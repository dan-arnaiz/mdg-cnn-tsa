from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

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

# Commented out assertions in case config.json doesn't match saved weights
# assert cfg["num_features"] == 32
# assert cfg["hidden_dim"] == 64
# assert cfg["num_heads"] == 2

class CNNTSA(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        # Transformer-style TSA block
        self.mhsa = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=2,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(64)

        # FIXED: Added Dropout layer (ffn.2) to match saved weights
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),      # ffn.0
            nn.ReLU(),                # ffn.1
            nn.Dropout(0.1),          # ffn.2
            nn.Linear(128, 64)        # ffn.3
        )
        self.norm2 = nn.LayerNorm(64)

        # Classifier
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 1, 32)
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
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=0,
            match=match,
            instructions=inst
        ))

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

            features = self.extract_features(flow)

            with torch.no_grad():
                pred = self.model(features).item()

            label = 1 if pred >= 0.5 else 0

            if label == 1:
                self.logger.warning("DDoS detected — blocking flow")
                self.block_flow(ev.msg.datapath, flow.match)

            self.log_result(features.numpy(), pred, ts)

    def extract_features(self, flow):
        dur = max(flow.duration_sec, 1)
        pkts = flow.packet_count
        bytes_ = flow.byte_count

        base = np.array([
            pkts,
            bytes_,
            dur,
            pkts / dur,
            bytes_ / dur,
            pkts / (bytes_ + 1),
            pkts / 1000,
            bytes_ / 1000
        ], dtype=np.float32)

        feat = np.zeros(32, dtype=np.float32)
        feat[:8] = base
        feat[8:16] = base
        feat[16:24] = base
        feat[24:32] = base[:8]
        return torch.tensor(feat).unsqueeze(0).unsqueeze(0)

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