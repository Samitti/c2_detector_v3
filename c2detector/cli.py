from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from .config import *
from .utils import banner, configure_logging
from .workflows import train_command, predict_command, predict_pcap_command, self_test_command

FEATURE_PROFILES = ["all-cleaned", "stable-core", "no-active-idle", "no-windows", "no-subflow"]

def build_parser():
    p=argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    sub=p.add_subparsers(dest="command",required=True)

    t=sub.add_parser("train", help="Train on CICIDS2017 and validate independently on CTU-13")
    t.add_argument("--cicids-dir",type=Path,default=DEFAULT_CICIDS_DIR)
    t.add_argument("--benign-file",default=DEFAULT_BENIGN_FILE)
    t.add_argument("--botnet-file",default=DEFAULT_BOTNET_FILE)
    t.add_argument("--ctu-file",type=Path,default=DEFAULT_CTU_FILE)
    t.add_argument("--ctu-all-malicious",action="store_true")
    t.add_argument("--feature-profile",choices=FEATURE_PROFILES,default="all-cleaned",
                   help="Feature group used for training and validation")
    t.add_argument("--test-size",type=float,default=.2)
    t.add_argument("--shap-sample",type=int,default=500)
    t.add_argument("--model-output",type=Path,default=DEFAULT_MODEL)
    t.add_argument("--report-output",type=Path,default=DEFAULT_TRAINING_REPORT)
    t.add_argument("--shap-output",type=Path,default=DEFAULT_SHAP_PLOT)
    t.add_argument("--metrics-csv",type=Path,default=DEFAULT_METRICS_CSV)
    t.add_argument("--confusion-plot",type=Path,default=DEFAULT_CONFUSION_PLOT)
    t.add_argument("--roc-plot",type=Path,default=DEFAULT_ROC_PLOT)
    t.add_argument("--feature-alignment-report",type=Path,default=DEFAULT_FEATURE_ALIGNMENT)
    t.add_argument("--distribution-report",type=Path,default=DEFAULT_DISTRIBUTION_REPORT)
    t.add_argument("--quality-report",type=Path,default=DEFAULT_QUALITY_REPORT)
    t.add_argument("--probability-report",type=Path,default=DEFAULT_PROBABILITY_REPORT)
    t.add_argument("--log-output",type=Path,default=DEFAULT_LOG)
    t.set_defaults(func=train_command)

    q=sub.add_parser("predict", help="Analyze a CICFlowMeter CSV using a saved model")
    q.add_argument("--input","-i",type=Path,required=True)
    q.add_argument("--model","-m",type=Path,default=DEFAULT_MODEL)
    q.add_argument("--output","-o",type=Path,default=DEFAULT_PREDICTION_REPORT)
    q.add_argument("--top-reasons",type=int,default=5)
    q.add_argument("--threshold",type=float,default=.5)
    q.add_argument("--log-output",type=Path,default=DEFAULT_LOG)
    q.set_defaults(func=predict_command)

    r=sub.add_parser("predict-pcap", help="Extract flows from PCAP, detect C2, and optionally enrich alerts with TLS metadata")
    r.add_argument("--input","-i",type=Path,required=True)
    r.add_argument("--model","-m",type=Path,default=DEFAULT_MODEL)
    r.add_argument("--output","-o",type=Path,default=None)
    r.add_argument("--flow-csv",type=Path,default=None)
    r.add_argument("--cicflowmeter-bin",default="cicflowmeter")
    r.add_argument("--overwrite-flow-csv",action="store_true")
    r.add_argument("--tls-enrich",action="store_true",help="Extract JA3/JA3S/SNI/cipher context with PyShark")
    r.add_argument("--tls-csv",type=Path,default=None)
    r.add_argument("--top-reasons",type=int,default=5)
    r.add_argument("--threshold",type=float,default=.5)
    r.add_argument("--log-output",type=Path,default=DEFAULT_LOG)
    r.set_defaults(func=predict_pcap_command)

    s=sub.add_parser("self-test")
    s.add_argument("--workdir",type=Path,default=Path("self_test"))
    s.add_argument("--log-output",type=Path,default=Path("self_test/output/logs/detector.log"))
    s.set_defaults(func=self_test_command)
    return p


def main():
    banner(); args=build_parser().parse_args(); configure_logging(getattr(args,"log_output",DEFAULT_LOG))
    logging.info("Started %s v%s command=%s",APP_NAME,APP_VERSION,args.command)
    try:
        if hasattr(args, "threshold") and not 0 < args.threshold < 1:
            raise ValueError("--threshold must be between 0 and 1.")
        return int(args.func(args))
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        print(f"\nERROR: {exc}",file=sys.stderr); return 2
    except KeyboardInterrupt:
        print("\nCancelled by user.",file=sys.stderr); return 130
