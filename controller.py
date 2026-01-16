from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet

import time
import os
import json
import numpy as np
import torch
import torch.nn as nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/corr1_k32/lr1e-05/main/best_weights.pt"
)
CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/corr1_k32/lr1e-05/main/config.json"
)
LOG_DIR = os.path.join(BASE_DIR, "merged_outputs")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

class CNNTSA(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor
        # FIXED: Changed from (1, 32) to (32, 32) to match saved weights
        self.conv1 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        # FIXED: Changed kernel_size from 5 to 3
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Transformer-style TSA block
        self.mhsa = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=2,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(64)

        # FFN block with Dropout
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),      # ffn.0.weight
            nn.ReLU(),                # activation
            nn.Dropout(0.1),          # ffn.2 (dropout layer)
            nn.Linear(128, 64)        # ffn.3.weight
        )
        self.norm2 = nn.LayerNorm(64)

        # Classifier
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 32, 32) - CHANGED from (B, 1, 32)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)

        attn_out, _ = self.mhsa(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Global average pooling
        x = x.mean(dim=1)

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# =======================
# RYU CONTROLLER
# =======================
class CNNTSAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        os.makedirs(LOG_DIR, exist_ok=True)

        try:
            self.model = CNNTSA()
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.logger.info("CNN-TSA model loaded successfully")

        except Exception as e:
            self.logger.error(f"MODEL LOAD FAILURE: {e}")
            raise RuntimeError("Model architecture mismatch")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp

        ofp = dp.ofproto
        parser = dp.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=0,
            match=match,
            instructions=inst
        ))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets and install forwarding rules"""
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match['in_port']

        # Parse packet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = dp.id

        # Learn MAC address to avoid flood
        self.mac_to_port = getattr(self, 'mac_to_port', {})
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofp.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow to avoid packet-in next time
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            
            dp.send_msg(parser.OFPFlowMod(
                datapath=dp,
                priority=1,
                match=match,
                instructions=inst,
                idle_timeout=60,
                hard_timeout=300
            ))

        # Send packet out
        data = None
        if msg.buffer_id == ofp.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        dp.send_msg(out)

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            hub.sleep(5)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        ts = time.time()

        for flow in ev.msg.body:
            if flow.priority == 0:
                continue

            # Pre-filter: Only analyze flows with significant traffic
            # This reduces false positives on normal ping/ARP traffic
            dur = max(flow.duration_sec, 1)
            pkts = flow.packet_count
            pkt_rate = pkts / dur
            
            # Skip flows that are clearly benign (low packet rate)
            # DDoS attacks typically have high packet rates (>100 pps)
            if pkt_rate < 50:  # Less than 50 packets/second = likely benign
                continue
            
            # Skip very short flows (less than 2 seconds)
            if dur < 2:
                continue

            features = self.extract_features(flow)

            with torch.no_grad():
                pred = self.model(features).item()

            label = 1 if pred >= 0.5 else 0

            if label == 1:
                self.logger.warning(f"DDoS detected (rate: {pkt_rate:.1f} pps, pred: {pred:.3f}) — blocking flow")
                self.block_flow(ev.msg.datapath, flow.match)

            self.log_result(features.numpy(), pred, ts)

    def extract_features(self, flow):
        """
        Extract 32 network flow features and reshape to (1, 32, 32)
        
        The model expects input shape: (batch_size, channels=32, sequence_length=32)
        """
        dur = max(flow.duration_sec, 1)
        pkts = max(flow.packet_count, 1)
        bytes_ = max(flow.byte_count, 1)
        
        # Calculate rates
        pkt_rate = pkts / dur
        byte_rate = bytes_ / dur
        avg_pkt_size = bytes_ / pkts
        
        # Create 32 diverse features (normalized)
        features = np.array([
            pkts / 10000.0,              # 0: Normalized packet count
            bytes_ / 1000000.0,          # 1: Normalized byte count  
            dur / 100.0,                 # 2: Normalized duration
            pkt_rate / 1000.0,           # 3: Packet rate
            byte_rate / 100000.0,        # 4: Byte rate
            avg_pkt_size / 1500.0,       # 5: Average packet size
            pkts / (bytes_ + 1),         # 6: Packet/byte ratio
            np.log1p(pkts),              # 7: Log packet count
            np.log1p(bytes_),            # 8: Log byte count
            np.log1p(pkt_rate),          # 9: Log packet rate
            np.log1p(byte_rate),         # 10: Log byte rate
            pkt_rate / (byte_rate + 1),  # 11: Rate ratio
            1.0 / (dur + 1),             # 12: Inverse duration
            1.0 / (avg_pkt_size + 1),    # 13: Inverse packet size
            np.sqrt(pkts),               # 14: Sqrt packet count
            np.sqrt(bytes_),             # 15: Sqrt byte count
            pkts ** 0.33,                # 16: Cube root packets
            bytes_ ** 0.33,              # 17: Cube root bytes
            pkt_rate ** 0.5,             # 18: Sqrt packet rate
            byte_rate ** 0.5,            # 19: Sqrt byte rate
            pkts * dur,                  # 20: Packet-duration product
            bytes_ * dur,                # 21: Byte-duration product
            pkts / 100.0,                # 22: Scaled packets
            bytes_ / 10000.0,            # 23: Scaled bytes
            pkt_rate / 100.0,            # 24: Scaled packet rate
            byte_rate / 10000.0,         # 25: Scaled byte rate
            avg_pkt_size / 100.0,        # 26: Scaled avg packet size
            (pkts + bytes_) / 10000.0,   # 27: Combined metric
            (pkt_rate + byte_rate) / 1000.0,  # 28: Combined rate
            np.tanh(pkt_rate / 100.0),   # 29: Tanh packet rate
            np.tanh(byte_rate / 1000.0), # 30: Tanh byte rate
            np.clip(avg_pkt_size / 1500.0, 0, 1)  # 31: Clipped packet size
        ], dtype=np.float32)
        
        # Reshape to (32, 32) by creating a window of features
        # Each row is slightly different to add temporal variation
        feat_matrix = np.zeros((32, 32), dtype=np.float32)
        for i in range(32):
            # Add small noise/variation to simulate temporal sequence
            noise = np.random.normal(0, 0.01, 32)
            feat_matrix[i] = features + noise * features
        
        # Convert to tensor with batch dimension: (1, 32, 32)
        return torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)

    def block_flow(self, dp, match):
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=100,
            match=match,
            instructions=[]
        ))

    def log_result(self, feat, pred, ts):
        with open(os.path.join(LOG_DIR, "detections.log"), "a") as f:
            f.write(f"{ts},{pred}\n")