from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import time
import os
import numpy as np
import joblib

MODEL_PATH = "models/cnn_tsa/model.pkl"
LOG_DIR = "merged_outputs"

class CNNTSAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CNNTSAController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        self.model = joblib.load(MODEL_PATH)
        self.logger.info("CNN–TSA Controller initialized")

    # ---------------------------------------------------
    # Switch connection
    # ---------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Table-miss rule
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=0,
            match=match,
            instructions=inst
        )
        datapath.send_msg(mod)

    # ---------------------------------------------------
    # Periodic monitoring
    # ---------------------------------------------------
    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(5)

    def request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ---------------------------------------------------
    # Flow stats handler
    # ---------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        timestamp = time.time()

        for flow in ev.msg.body:
            if flow.priority == 0:
                continue

            features = self.extract_features(flow)
            prediction = self.model.predict([features])[0]

            if prediction == 1:
                self.logger.warning("DDoS detected — blocking flow")
                self.block_flow(ev.msg.datapath, flow.match)

            self.log_result(features, prediction, timestamp)

    # ---------------------------------------------------
    # Feature extraction (CICDDoS2019-aligned)
    # ---------------------------------------------------
    def extract_features(self, flow):
        duration = max(flow.duration_sec, 1)
        packets = flow.packet_count
        bytes_ = flow.byte_count
        pps = packets / duration

        return np.array([
            packets,
            bytes_,
            duration,
            pps
        ])

    # ---------------------------------------------------
    # Mitigation
    # ---------------------------------------------------
    def block_flow(self, datapath, match):
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=100,
            match=match,
            instructions=[]
        )
        datapath.send_msg(mod)

    # ---------------------------------------------------
    # Logging (for Section 3.7)
    # ---------------------------------------------------
    def log_result(self, features, prediction, timestamp):
        with open(f"{LOG_DIR}/detections.log", "a") as f:
            f.write(f"{timestamp},{features.tolist()},{prediction}\n")
