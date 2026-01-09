from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import time
import os
import numpy as np
import torch  # Replaced joblib with torch

# Updated path to point to your best_weights.pt
MODEL_PATH = "models/cnn_tsa/corr1_k32/lr1e-05/main/best_weights.pt"
LOG_DIR = "merged_outputs"

class CNNTSAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CNNTSAController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # Load PyTorch model on CPU
        try:
            # We use map_location='cpu' to ensure it loads even if trained on a GPU
            self.model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
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

        # Create numpy array and convert to PyTorch Tensor
        feat_array = np.array([packets, bytes_, duration, pps], dtype=np.float32)
        # Add batch dimension (1, 4) as CNNs expect a batch
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