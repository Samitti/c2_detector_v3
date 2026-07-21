from pathlib import Path

APP_NAME = "C2 Traffic Detector"
APP_VERSION = "3.2"
RANDOM_STATE = 42

DEFAULT_CICIDS_DIR = Path("data/cicids2017/MachineLearningCVE")
DEFAULT_BENIGN_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
DEFAULT_BOTNET_FILE = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
DEFAULT_CTU_FILE = Path("data/ctu13/ctu13_flows/botnet-capture-20110810-neris.pcap_Flow.csv")

OUTPUT_ROOT = Path("output")
DEFAULT_MODEL = OUTPUT_ROOT / "models/c2_model.json"
DEFAULT_TRAINING_REPORT = OUTPUT_ROOT / "reports/training_report.json"
DEFAULT_SHAP_PLOT = OUTPUT_ROOT / "shap/shap_summary.png"
DEFAULT_PREDICTION_REPORT = OUTPUT_ROOT / "reports/forensic_report.json"
DEFAULT_FLOW_DIR = OUTPUT_ROOT / "flows"
DEFAULT_TLS_DIR = OUTPUT_ROOT / "tls"
DEFAULT_METRICS_CSV = OUTPUT_ROOT / "validation/metrics.csv"
DEFAULT_CONFUSION_PLOT = OUTPUT_ROOT / "validation/confusion_matrix.png"
DEFAULT_ROC_PLOT = OUTPUT_ROOT / "validation/roc_curve.png"
DEFAULT_FEATURE_ALIGNMENT = OUTPUT_ROOT / "validation/feature_alignment_report.csv"
DEFAULT_DISTRIBUTION_REPORT = OUTPUT_ROOT / "validation/feature_distribution_report.csv"
DEFAULT_QUALITY_REPORT = OUTPUT_ROOT / "validation/data_quality_report.csv"
DEFAULT_PROBABILITY_REPORT = OUTPUT_ROOT / "validation/ctu_probability_distribution.csv"
DEFAULT_ABLATION_REPORT = OUTPUT_ROOT / "validation/ablation_results.csv"
DEFAULT_LOG = OUTPUT_ROOT / "logs/detector.log"

IDENTIFIER_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP",
    "Src Port", "Source Port", "Protocol",
]

COLUMN_MAPPING = {
    "Dst Port": "Destination Port",
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

DUPLICATE_FEATURES = {"Fwd Header Length.1"}
SUBFLOW_FEATURES = {"Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes"}
ACTIVE_IDLE_FEATURES = {
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
}
WINDOW_FEATURES = {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}

# Features that should not be negative. TCP initial-window values are intentionally excluded
# because -1 can be a documented sentinel in CICFlowMeter-derived datasets.
NONNEGATIVE_EXACT = {
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Header Length", "Bwd Header Length", "min_seg_size_forward",
    "act_data_pkt_fwd", "Down/Up Ratio",
}
NONNEGATIVE_KEYWORDS = (
    "Packet Length", "Packets/s", "Bytes/s", "IAT", "Flag Count", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Active ", "Idle ", "Subflow ",
)

STABLE_CORE_FEATURES = {
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "act_data_pkt_fwd", "min_seg_size_forward",
}
