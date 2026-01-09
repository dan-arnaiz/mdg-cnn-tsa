from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import time
import os
import json
import numpy as np
import torch  # Replaced joblib with torch

# Updated path to point to your best_weights.pt (use absolute paths for ryu-manager compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/cnn_tsa/corr1_k32/lr1e-05/main/best_weights.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "models/cnn_tsa/corr1_k32/lr1e-05/main/config.json")
LOG_DIR = os.path.join(BASE_DIR, "merged_outputs")


# Define your CNN-TSA model architecture (adjust if different from your actual model)
class CNNTSA(torch.nn.Module):
    def __init__(self, num_features, num_heads, hidden_dim, dropout=0.1):
        super(CNNTSA, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)
        self.attention = torch.nn.MultiheadAttention(32, num_heads, batch_first=True)
        self.fc1 = torch.nn.Linear(32, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, num_features) or (batch_size, 1, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # Prepare for attention
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.mean(dim=1)  # Global average pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class CNNTSAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CNNTSAController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # Load configuration and model
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            # Create model with the architecture from config
            self.model = CNNTSA(
                num_features=config['num_features'],
                num_heads=config['num_heads'],
                hidden_dim=config['hidden_dim'],
                dropout=config['regularization']['cnn_dropout']
            )
            
            # Load the saved weights (state_dict)
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set model to evaluation mode for inference
            self.logger.info("CNN–TSA Controller initialized with PyTorch model")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=0,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(5)

    def request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        timestamp = time.time()

        for flow in ev.msg.body:
            if flow.priority == 0:
                continue

            # Get features as a Torch Tensor
            features_tensor = self.extract_features(flow)
            
            # Perform inference without tracking gradients
            with torch.no_grad():
                output = self.model(features_tensor)
                # Assuming binary classification output (e.g., sigmoid or argmax)
                prediction = torch.round(torch.sigmoid(output)).item() if output.dim() <= 1 else torch.argmax(output, dim=1).item()

            if prediction == 1:
                self.logger.warning("DDoS detected — blocking flow")
                self.block_flow(ev.msg.datapath, flow.match)

            self.log_result(features_tensor.numpy(), prediction, timestamp)

    def extract_features(self, flow):
        duration = max(flow.duration_sec, 1)
        packets = flow.packet_count
        bytes_ = flow.byte_count
        pps = packets / duration
        bps = (bytes_ * 8) / duration  # bits per second
        
        # Extract 8 base features
        base_features = np.array([
            packets, bytes_, duration, pps, bps,
            packets / (duration * 1000) if duration > 0 else 0,  # packets per ms
            bytes_ / (duration * 1000) if duration > 0 else 0,   # bytes per ms
            packets / (bytes_ + 1)  # packet to byte ratio (avoid division by 0)
        ], dtype=np.float32)
        
        # Pad to 32 features (required by model) with repeated statistics
        feat_array = np.zeros(32, dtype=np.float32)
        feat_array[:8] = base_features
        feat_array[8:16] = base_features  # Repeat basic stats
        feat_array[16:24] = base_features
        feat_array[24:32] = base_features
        
        # Add batch dimension (1, 32) as CNNs expect batch and sequential data
        return torch.from_numpy(feat_array).unsqueeze(0)

    def block_flow(self, datapath, match):
        parser = datapath.ofproto_parser
        mod = parser.OFPFlowMod(datapath=datapath, priority=100,
                                match=match, instructions=[])
        datapath.send_msg(mod)

    def log_result(self, features, prediction, timestamp):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        with open(f"{LOG_DIR}/detections.log", "a") as f:
            f.write(f"{timestamp},{features.tolist()},{prediction}\n")