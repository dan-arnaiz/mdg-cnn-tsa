from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet

import time
import os
import json
import math
import urllib.request
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "merged_outputs")

ML_SERVER_URL       = "http://127.0.0.1:5000/predict"
DETECTION_THRESHOLD = 0.50


# =======================
# RYU CONTROLLER
# =======================
class CNNTSAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.simulation_start = time.time()
        self.datapaths        = {}
        self.mac_to_port      = {}
        self.monitor_thread   = hub.spawn(self.monitor)

        os.makedirs(LOG_DIR, exist_ok=True)

        # Verify ML server is reachable at startup
        try:
            req = urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=5)
            if req.status == 200:
                self.logger.info("ML inference server reachable at localhost:5000")
        except Exception as e:
            self.logger.warning(f"ML server not reachable yet: {e} — will retry per flow")

    # ── Entropy helper ────────────────────────────────────────────────────────
    def calculate_entropy(self, src_list):
        if not src_list:
            return 0
        counts  = Counter(src_list)
        total   = len(src_list)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    # ── OpenFlow handlers ─────────────────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp     = ev.msg.datapath
        ofp    = dp.ofproto
        parser = dp.ofproto_parser

        self.datapaths[dp.id] = dp

        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst    = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        dp.send_msg(parser.OFPFlowMod(
            datapath=dp, priority=0, match=match, instructions=inst
        ))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg     = ev.msg
        dp      = msg.datapath
        ofp     = dp.ofproto
        parser  = dp.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst  = eth.dst
        src  = eth.src
        dpid = dp.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions  = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            inst  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            dp.send_msg(parser.OFPFlowMod(
                datapath=dp, priority=1, match=match, instructions=inst,
                idle_timeout=60, hard_timeout=300
            ))

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        ))

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            hub.sleep(1)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        ts             = time.time()
        total_flows    = 0
        analyzed_flows = 0
        total_packets_in_cycle = 0

        # First pass: entropy calculation
        src_list = []
        for flow in ev.msg.body:
            if flow.priority == 0:
                continue
            total_packets_in_cycle += flow.packet_count
            if 'eth_src' in flow.match:
                src_list.append(flow.match.get('eth_src'))

        entropy           = self.calculate_entropy(src_list)
        dynamic_threshold = 3.0 if entropy > 1.2 else 6.0

        self.logger.info(
            f"Entropy: {entropy:.2f} | Total Pkts: {total_packets_in_cycle} "
            f"| Threshold: {dynamic_threshold}"
        )

        # Second pass: inference on qualifying flows
        for flow in ev.msg.body:
            if flow.priority == 0:
                continue

            total_flows += 1
            dur      = max(flow.duration_sec, 1)
            pkts     = flow.packet_count
            pkt_rate = pkts / dur

            if pkt_rate < dynamic_threshold or dur < 1:
                continue

            analyzed_flows += 1

            try:
                features = self.extract_raw_features(flow)
                pred     = self.call_ml_server(features)

                if pred is None:
                    continue

                label = 1 if pred >= DETECTION_THRESHOLD else 0

                if label == 1:
                    self.logger.warning(
                        f"DDoS detected (score={pred:.3f}, rate={pkt_rate:.1f} pps) — blocking"
                    )
                    self.block_flow(ev.msg.datapath, flow.match)

                self.log_result(pred, label, ts, pkt_rate)

            except Exception as e:
                self.logger.error(f"Flow processing error: {e}")
                continue

        if analyzed_flows > 0:
            self.logger.info(f"Analyzed {analyzed_flows}/{total_flows} flows")

    # ── ML server call ────────────────────────────────────────────────────────
    def call_ml_server(self, raw_feature_dict):
        """Send features to ML inference server, return probability or None on failure."""
        try:
            payload = json.dumps({"features": raw_feature_dict}).encode("utf-8")
            req     = urllib.request.Request(
                ML_SERVER_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("prediction")
        except Exception as e:
            self.logger.error(f"ML server call failed: {e}")
            return None

    # ── Feature extraction ────────────────────────────────────────────────────
    def extract_raw_features(self, flow):
        """
        Returns a raw feature dict with EXACT column names matching the training
        preprocessor. Keys match selected_features in preprocess_metadata.json
        (without the 'num__' prefix, added internally by ColumnTransformer).
        """
        dur_us  = max(flow.duration_sec * 1e6, 1)   # CICFlowMeter uses microseconds
        dur_sec = max(flow.duration_sec, 1e-6)
        pkts    = max(flow.packet_count, 1)
        bytes_  = max(flow.byte_count, 1)

        flow_bytes_per_s = bytes_ / dur_sec
        flow_pkts_per_s  = pkts   / dur_sec
        avg_pkt_size     = bytes_ / pkts
        fwd_header_len   = 20   # Min IPv4+TCP header bytes

        return {
            "ACK Flag Count":           0.0,
            "CWE Flag Count":           0.0,
            "SYN Flag Count":           0.0,
            "URG Flag Count":           0.0,
            "Fwd PSH Flags":            0.0,
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
            "Fwd Header Length":        float(fwd_header_len * pkts),
            "Bwd Header Length":        float(fwd_header_len),
            "Flow Bytes/s":             flow_bytes_per_s,
            "Flow Packets/s":           flow_pkts_per_s,
            "Bwd Packets/s":            0.0,
            "Subflow Fwd Packets":      float(pkts),
            "Subflow Fwd Bytes":        float(bytes_),
            "Subflow Bwd Bytes":        0.0,
            "Init_Win_bytes_forward":   65535.0,
            "Init_Win_bytes_backward":  65535.0,
            "Active Max":               0.0,
            "Active Mean":              0.0,
            "Active Std":               0.0,
            "Idle Std":                 0.0,
            "Down/Up Ratio":            0.0,
            "Protocol":                 6.0,
            "act_data_pkt_fwd":         float(pkts),
            "min_seg_size_forward":     20.0,
        }

    # ── Flow blocking ─────────────────────────────────────────────────────────
    def block_flow(self, dp, match):
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=100,
            match=match,
            idle_timeout=30,
            instructions=[]
        ))

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_result(self, pred, label, ts, pkt_rate):
        flag_path  = os.path.join(BASE_DIR, "merged_outputs", "attack_started.flag")
        true_label = 1 if os.path.exists(flag_path) else 0

        with open(os.path.join(LOG_DIR, "detections.log"), "a") as f:
            f.write(f"{ts},{pred:.6f},{label},{true_label},{pkt_rate:.2f}\n")