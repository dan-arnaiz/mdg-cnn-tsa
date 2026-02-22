from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet

import joblib
import time
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from collections import Counter



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/baseline_model/main/standard_k45/best_weights.pt"
)
CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/baseline_model/main/standard_k45/config.json"
)
LOG_DIR = os.path.join(BASE_DIR, "merged_outputs")

# Load configuration
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

print(f"Configuration loaded: num_features={cfg['num_features']}, "
      f"num_heads={cfg['num_heads']}, hidden_dim={cfg['hidden_dim']}")


class CNNTSA(nn.Module):
    def __init__(self, num_features=39, hidden_dim=64, num_heads=2):
        super().__init__()

        # CNN feature extractor - uses num_features as input channels
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Transformer-style TSA block
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # FFN block with Dropout
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, num_features, T)
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

    # Load trained model
    model = torch.load("models/cnn_tsa/baseline_model/main/standard_k45/best_weights.pt")
    model.eval()

    # Load preprocessing artifacts
    preprocessor = joblib.load("preprocessing_output/v1_std_corr90_k45_w48s24/preprocessor.joblib")
    selector = joblib.load("preprocessing_output/v1_std_corr90_k45_w48s24/selector.joblib")

    # Load metadata to ensure correct feature names
    with open("preprocessing_output/v1_std_corr90_k45_w48s24/preprocess_metadata.json") as f:
        metadata = json.load(f)

    selected_features = metadata["selected_features"]
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.simulation_start = time.time()
        self.attack_start_delay = 30  # Seconds before attack traffic starts
        
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)
        self.mac_to_port = {}

        os.makedirs(LOG_DIR, exist_ok=True)

        try:
            # Initialize model with config parameters
            self.model = CNNTSA(
                num_features=cfg['num_features'],
                hidden_dim=cfg['hidden_dim'],
                num_heads=cfg['num_heads']
            )
            
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            self.logger.info(f"CNN-TSA model loaded successfully")
            self.logger.info(f"Model config: features={cfg['num_features']}, "
                           f"hidden={cfg['hidden_dim']}, heads={cfg['num_heads']}")

        except Exception as e:
            self.logger.error(f"MODEL LOAD FAILURE: {e}")
            raise RuntimeError("Model architecture mismatch")
        
    # ADDED: Entropy calculation method to identify spoofing
    def calculate_entropy(self, src_list):
        if not src_list:
            return 0
        counts = Counter(src_list)
        total = len(src_list)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    
    def preprocess_live_features(self, raw_feature_dict):
        """
        Preprocesses a raw feature dict through the training pipeline.
        The preprocessor expects column names WITHOUT the 'num__' prefix
        (that prefix is added by sklearn's ColumnTransformer internally).
        """
        import pandas as pd
        df = pd.DataFrame([raw_feature_dict])

        # Apply training preprocessor (StandardScaler + ColumnTransformer)
        Xt = self.preprocessor.transform(df)

        # Convert sparse to dense if needed
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        # Apply SelectKBest (k=45 → then reduced to 39 after corr drop)
        Xt_sel = self.selector.transform(Xt)

        return Xt_sel  # shape: (1, 39)

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

        # Learn MAC address
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
            hub.sleep(1) # ADJUSTED from 2s to 1s to catch bursts faster

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        ts = time.time()
        total_flows = 0
        analyzed_flows = 0
        total_packets_in_cycle = 0 # ADDED: Track total packets in this batch

        # Extract all source MACs and sum total packets
        src_list = []
        for flow in ev.msg.body:
            if flow.priority == 0:
                continue
            
            total_packets_in_cycle += flow.packet_count # ADDED: Accumulate packet count
            
            if 'eth_src' in flow.match:
                src_list.append(flow.match.get('eth_src'))

        # Calculate entropy based on the collected source list
        entropy = self.calculate_entropy(src_list)

        # UPDATED: Dynamic Threshold to avoid inflated recall.
        dynamic_threshold = 3.0 if entropy > 1.2 else 6.0
        
        # UPDATED: Log both Entropy and Total Packets for debugging
        self.logger.info(f"Entropy: {entropy:.2f} | Total Pkts: {total_packets_in_cycle} | Threshold: {dynamic_threshold}")

        for flow in ev.msg.body:
            if flow.priority == 0:
                continue
            
            total_flows += 1
            dur = max(flow.duration_sec, 1)
            pkts = flow.packet_count
            pkt_rate = pkts / dur

            # Sensitivity filter
            if pkt_rate < dynamic_threshold:
                continue
            if dur < 1:
                continue

            raw_feature_dict = self.extract_raw_features(flow)
            Xt_live = self.preprocess_live_features(raw_feature_dict)

            # Model expects (batch, num_features, sequence_length)
            # We have (1, 39) → expand to (1, 39, 1)
            Xt_tensor = torch.tensor(Xt_live, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                output = self.model(Xt_tensor)
                pred = output.item()  # sigmoid already applied in model.forward()

            DETECTION_THRESHOLD = 0.50
            label = 1 if pred >= DETECTION_THRESHOLD else 0
            if label == 1:
                self.logger.warning(f"DDoS detected (rate: {pkt_rate:.1f} pps) — blocking flow")
                self.block_flow(ev.msg.datapath, flow.match)

            self.log_result(pred, label, ts, pkt_rate)

        if analyzed_flows > 0:
            self.logger.info(f"Analyzed {analyzed_flows}/{total_flows} flows")

    def extract_raw_features(self, flow):
        """
        Returns a raw feature dict with EXACT column names matching the training
        preprocessor. Keys match selected_features in preprocess_metadata.json
        (without the 'num__' prefix, which is added internally by ColumnTransformer).
        """
        dur_us  = max(flow.duration_sec * 1e6, 1)   # CICFlowMeter uses microseconds
        dur_sec = max(flow.duration_sec, 1e-6)
        pkts    = max(flow.packet_count, 1)
        bytes_  = max(flow.byte_count, 1)

        flow_bytes_per_s = bytes_ / dur_sec
        flow_pkts_per_s  = pkts   / dur_sec
        avg_pkt_size     = bytes_ / pkts
        fwd_header_len   = 20   # Min IPv4+TCP header in bytes

        return {
            "ACK Flag Count":           0,
            "CWE Flag Count":           0,
            "SYN Flag Count":           0,
            "URG Flag Count":           0,
            "Fwd PSH Flags":            0,
            "Average Packet Size":      avg_pkt_size,
            "Max Packet Length":        avg_pkt_size,
            "Fwd Packet Length Max":    avg_pkt_size,
            "Fwd Packet Length Std":    0.0,
            "Bwd Packet Length Max":    0.0,
            "Bwd Packet Length Min":    0.0,
            "Avg Bwd Segment Size":     0.0,
            "Flow Duration":            dur_us,
            "Flow IAT Max":             dur_us,
            "Flow IAT Mean":            dur_us / max(pkts - 1, 1),
            "Flow IAT Min":             0.0,
            "Flow IAT Std":             0.0,
            "Bwd IAT Total":            0.0,
            "Bwd IAT Max":              0.0,
            "Bwd IAT Mean":             0.0,
            "Bwd IAT Min":              0.0,
            "Fwd Header Length":        fwd_header_len * pkts,
            "Bwd Header Length":        float(fwd_header_len),
            "Flow Bytes/s":             flow_bytes_per_s,
            "Flow Packets/s":           flow_pkts_per_s,
            "Bwd Packets/s":            0.0,
            "Subflow Fwd Packets":      pkts,
            "Subflow Fwd Bytes":        bytes_,
            "Subflow Bwd Bytes":        0,
            "Init_Win_bytes_forward":   65535,
            "Init_Win_bytes_backward":  65535,
            "Active Max":               0.0,
            "Active Mean":              0.0,
            "Active Std":               0.0,
            "Idle Std":                 0.0,
            "Down/Up Ratio":            0.0,
            "Protocol":                 6,
            "act_data_pkt_fwd":         pkts,
            "min_seg_size_forward":     20,
        }

    def block_flow(self, dp, match):
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=100,
            match=match,
            idle_timeout=30,
            instructions=[]
        ))

    def log_result(self, pred, label, ts, pkt_rate):

        # TRUE label based on attack marker
        if os.path.exists("merged_outputs/attack_started.flag"):
            true_label = 1
        else:
            true_label = 0

        with open(os.path.join(LOG_DIR, "detections.log"), "a") as f:
            f.write(f"{ts},{pred:.6f},{label},{true_label},{pkt_rate:.2f}\n")