from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, shap, xgboost as xgb
from sklearn.model_selection import train_test_split
from .config import *
from .data import *
from .modeling import *
from .explain import *
from .reporting import save_validation_artifacts, save_dataframe
from .pcap import run_cicflowmeter
from .tls import extract_tls_metadata, enrich_alerts_with_tls
from .utils import step, ensure_parent, utc_now

def external_validate(model, ctu_file, feature_names, training_reference, *, ctu_all_malicious,
                      alignment_report_path, distribution_report_path, probability_report_path,
                      feature_profile="all-cleaned"):

    if ctu_file is None or not ctu_file.exists():
        print("  CTU-13 validation      : skipped (file not found)")
        return None
    raw = read_csv_checked(ctu_file)
    rawX, y, label_info = extract_labels(raw, source_name=str(ctu_file), all_malicious=ctu_all_malicious)
    # Preserve constant external columns. Removing them would falsely turn them into missing features.
    source_X, _, cleaning = numeric_features(rawX, remove_constant=False, feature_profile=feature_profile)
    alignment_table = build_feature_alignment_report(source_X, feature_names)
    save_dataframe(alignment_table, alignment_report_path)
    X, alignment = align_features(source_X, feature_names)
    distribution_table = build_distribution_report(training_reference, X, feature_names)
    save_dataframe(distribution_table, distribution_report_path)
    shift_count = int(distribution_table["shift_warning"].sum())
    alignment["features_with_large_distribution_shift"] = shift_count
    alignment["alignment_report"] = str(alignment_report_path)
    alignment["distribution_report"] = str(distribution_report_path)
    print(f"  Training features      : {len(feature_names):,}")
    print(f"  CTU source features    : {source_X.shape[1]:,}")
    print(f"  Matched features       : {alignment['matched_feature_count']:,}")
    print(f"  Missing features       : {len(alignment['missing_features_filled_with_zero']):,}")
    print(f"  Extra features         : {len(alignment['extra_features_dropped']):,}")
    print(f"  Feature coverage       : {alignment['coverage_ratio']:.2%}")
    print(f"  Large shift warnings   : {shift_count:,}")
    if alignment["coverage_ratio"] < 0.90:
        print("  WARNING                : Feature coverage is below 90%; external results may be unreliable.")
    if shift_count:
        print("  WARNING                : Strong dataset shift detected; review the distribution report.")
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    prob_rows = [{"statistic": "minimum", "value": float(np.min(prob))},
                 {"statistic": "median", "value": float(np.median(prob))},
                 {"statistic": "mean", "value": float(np.mean(prob))},
                 {"statistic": "maximum", "value": float(np.max(prob))}]
    prob_rows += [{"statistic": f"flows_at_or_above_{t:.2f}", "value": int((prob >= t).sum())} for t in thresholds]
    save_dataframe(pd.DataFrame(prob_rows), probability_report_path)
    print(f"  C2 probability min     : {float(np.min(prob)):.6f}")
    print(f"  C2 probability median  : {float(np.median(prob)):.6f}")
    print(f"  C2 probability max     : {float(np.max(prob)):.6f}")
    print(f"  Flows >= 0.25          : {int((prob >= .25).sum()):,}")
    print(f"  Flows >= 0.50          : {int((prob >= .50).sum()):,}")
    m = compute_metrics(y, pred, prob)
    m.update({"labeling": label_info, "cleaning": cleaning, "feature_alignment": alignment,
              "dataset": str(ctu_file), "probability_distribution_report": str(probability_report_path)})
    return m

def train_command(args):
    step("STEP 1 — Loading CICIDS2017 training data")
    benign_path=args.cicids_dir/args.benign_file; botnet_path=args.cicids_dir/args.botnet_file
    bX,by,binfo=extract_labels(read_csv_checked(benign_path),source_name=str(benign_path)); bX=bX.loc[by==0].reset_index(drop=True); by=by.loc[by==0].reset_index(drop=True)
    mX,my,minfo=extract_labels(read_csv_checked(botnet_path),source_name=str(botnet_path)); mX=mX.loc[my==1].reset_index(drop=True); my=my.loc[my==1].reset_index(drop=True)
    if len(bX)==0 or len(mX)==0: raise ValueError("Could not find both benign and Bot-labeled CICIDS2017 rows.")
    X,_,cleaning=numeric_features(pd.concat([bX,mX],ignore_index=True,sort=False), remove_constant=True, feature_profile=args.feature_profile); y=pd.concat([by,my],ignore_index=True)
    print(f"  CICIDS benign rows     : {int((y==0).sum()):,}\n  CICIDS C2 rows         : {int((y==1).sum()):,}\n  Numeric features       : {X.shape[1]:,}")
    step("STEP 2 — Creating independent CICIDS2017 train/test split"); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=args.test_size,random_state=RANDOM_STATE,stratify=y)
    step("STEP 3 — Training XGBoost"); model=build_model(ytr); model.fit(Xtr,ytr); print("  Model trained successfully")
    step("STEP 4 — Internal CICIDS2017 evaluation"); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]; internal=compute_metrics(yte,pred,prob); print_metrics("CICIDS2017 held-out test set",internal); save_validation_artifacts(yte,pred,prob,internal,args.metrics_csv,args.confusion_plot,args.roc_plot)
    step("STEP 5 — Independent CTU-13 external validation"); external=external_validate(model,args.ctu_file,X.columns.tolist(),Xtr,ctu_all_malicious=args.ctu_all_malicious,alignment_report_path=args.feature_alignment_report,distribution_report_path=args.distribution_report,probability_report_path=args.probability_report,feature_profile=args.feature_profile); print_metrics("CTU-13 external validation",external) if external else None
    step("STEP 6 — Generating global SHAP summary"); top=create_global_shap_plot(model,Xte,args.shap_output,args.shap_sample)
    step("STEP 7 — Saving model and report"); ensure_parent(args.model_output); model.save_model(args.model_output)
    report={"report_metadata":{"generated_at_utc":utc_now(),"tool":f"{APP_NAME} v{APP_VERSION}","purpose":"Research evaluation of encrypted C2 flow detection"},"data_provenance":{"cicids_benign_file":str(benign_path),"cicids_botnet_file":str(botnet_path),"ctu13_file":str(args.ctu_file),"benign_labeling":binfo,"botnet_labeling":minfo,"feature_count":len(X.columns),"feature_names":X.columns.tolist(),"feature_profile":args.feature_profile,"cleaning":cleaning},"internal_cicids2017_evaluation":internal,"external_ctu13_evaluation":external,"global_explainability":{"method":"SHAP TreeExplainer","plot":str(args.shap_output),"top_features":top},"limitations":["Research prototype only; not operational certification.","External FPR cannot be measured when CTU-13 is treated as malicious-only."],"output_files":{"model":str(args.model_output),"training_report":str(args.report_output),"feature_alignment_report":str(args.feature_alignment_report),"feature_distribution_report":str(args.distribution_report),"ctu_probability_report":str(args.probability_report),"data_quality_report":str(args.quality_report)}}
    quality_rows = cleaning.get("quality_rows", [])
    save_dataframe(pd.DataFrame(quality_rows or [{"feature":"none","invalid_negative_or_range_values":0}]), args.quality_report)
    ensure_parent(args.report_output); args.report_output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"  Model saved            : {args.model_output}\n  Report saved           : {args.report_output}"); return 0

def predict_command(args):
    if not args.model.exists(): raise FileNotFoundError(f"Saved model not found: {args.model}")
    step("PREDICT 1 — Loading model and traffic CSV"); model=xgb.XGBClassifier(); model.load_model(args.model); features=model.get_booster().feature_names
    raw=read_csv_checked(args.input); Xraw,ids,cleaning=numeric_features(normalize_columns(raw), remove_constant=False, feature_profile="all-cleaned"); X,alignment=align_features(Xraw,features); print(f"  Flows loaded           : {len(X):,}")
    step("PREDICT 2 — Detecting C2 flows"); probs=model.predict_proba(X)[:,1]; preds=(probs>=args.threshold).astype(int); idxs=np.flatnonzero(preds==1); print(f"  C2 alerts              : {len(idxs):,}")
    alerts=[]
    if len(idxs):
        alertX=X.iloc[idxs]; explainer=shap.TreeExplainer(model); vals=shap_matrix(explainer,alertX)
        for pos,row in enumerate(idxs):
            identity={k:(v.item() if isinstance(v,(np.integer,np.floating)) else v) for k,v in ids.iloc[row].dropna().to_dict().items()} if not ids.empty else {}
            alerts.append({"event_id":f"c2-{int(row):08d}","source_row":int(row),"event_type":"encrypted_c2_suspected","severity":"high" if probs[row]>=0.9 else "medium","verdict":"C2_MALICIOUS","confidence":round(float(probs[row]),6),"threshold":args.threshold,"network_identifiers":identity,"explanation_method":"SHAP TreeExplainer","top_reasons":top_shap_reasons(features,X.iloc[row].to_numpy(float),vals[pos],args.top_reasons)})
    tls_enrichment = getattr(args, "tls_enrichment", None)
    if tls_enrichment and tls_enrichment.get("dataframe") is not None:
        tls_summary = enrich_alerts_with_tls(alerts, tls_enrichment["dataframe"])
        tls_summary["protocol_coverage"] = tls_enrichment.get("coverage", {})
    else:
        tls_summary = None
    report={"report_metadata":{"generated_at_utc":utc_now(),"tool":f"{APP_NAME} v{APP_VERSION}","input_file":str(args.input),"source_pcap":str(getattr(args,"source_pcap","")) or None,"flow_extractor":getattr(args,"flow_extractor",None),"model_used":str(args.model)},"summary":{"total_flows_analyzed":len(X),"c2_alerts_raised":len(alerts),"alert_rate":round(len(alerts)/len(X),8) if len(X) else 0.0,"decision_threshold":args.threshold},"data_quality":{"cleaning":cleaning,"feature_alignment":alignment},"tls_enrichment":tls_summary,"alerts":alerts,"limitations":["An alert is an investigative lead, not proof of criminal activity."]}
    ensure_parent(args.output); args.output.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8"); print(f"  Report saved           : {args.output}"); return 0

def predict_pcap_command(args):
    step("PCAP 1 — Converting packet capture to CICFlowMeter CSV"); pcap=args.input.resolve(); csv=args.flow_csv or (DEFAULT_FLOW_DIR/f"{pcap.stem}_Flow.csv"); info=run_cicflowmeter(pcap,csv,args.cicflowmeter_bin,overwrite=args.overwrite_flow_csv); out=args.output or (OUTPUT_ROOT/"reports"/f"{pcap.stem}_forensic_report.json")
    tls_enrichment = None
    if args.tls_enrich:
        step("PCAP 2 — Extracting TLS forensic metadata")
        tls_df, coverage = extract_tls_metadata(pcap)
        tls_path = args.tls_csv or (DEFAULT_TLS_DIR / f"{pcap.stem}_tls.csv")
        ensure_parent(tls_path); tls_df.to_csv(tls_path, index=False)
        print(f"  TLS records extracted  : {len(tls_df):,}")
        print(f"  TLS metadata saved     : {tls_path}")
        tls_enrichment = {"dataframe": tls_df, "coverage": coverage, "csv": str(tls_path)}
    return predict_command(argparse.Namespace(input=csv,model=args.model,output=out,top_reasons=args.top_reasons,threshold=args.threshold,source_pcap=pcap,flow_extractor=info,tls_enrichment=tls_enrichment))

def self_test_command(args):
    rng=np.random.default_rng(RANDOM_STATE); work=args.workdir; cic=work/"data/cicids2017/MachineLearningCVE"; ctu=work/"data/ctu13/ctu13_flows"; cic.mkdir(parents=True,exist_ok=True); ctu.mkdir(parents=True,exist_ok=True)
    benign=pd.DataFrame({"Flow Duration":rng.normal(1000,120,240),"Fwd Header Length":rng.normal(32,3,240),"Destination Port":rng.choice([80,443,53],240),"Flow IAT Min":rng.normal(120,15,240),"Label":"BENIGN"}); bot=pd.DataFrame({"Flow Duration":rng.normal(120,25,90),"Fwd Header Length":rng.normal(60,5,90),"Destination Port":rng.choice([4444,8081,9001],90),"Flow IAT Min":rng.normal(8,2,90),"Label":"Bot"})
    benign.to_csv(cic/DEFAULT_BENIGN_FILE,index=False); pd.concat([bot,benign.sample(30,random_state=1)]).to_csv(cic/DEFAULT_BOTNET_FILE,index=False); pd.concat([benign.sample(50,random_state=2),bot.sample(50,random_state=3)]).rename(columns={"Destination Port":"Dst Port"}).to_csv(ctu/DEFAULT_CTU_FILE.name,index=False)
    train_command(argparse.Namespace(cicids_dir=cic,benign_file=DEFAULT_BENIGN_FILE,botnet_file=DEFAULT_BOTNET_FILE,ctu_file=ctu/DEFAULT_CTU_FILE.name,ctu_all_malicious=False,test_size=.2,shap_sample=50,model_output=work/"output/models/c2_model.json",report_output=work/"output/reports/training_report.json",shap_output=work/"output/shap/shap_summary.png",metrics_csv=work/"output/validation/metrics.csv",confusion_plot=work/"output/validation/confusion_matrix.png",roc_plot=work/"output/validation/roc_curve.png",feature_alignment_report=work/"output/validation/feature_alignment_report.csv",distribution_report=work/"output/validation/feature_distribution_report.csv",probability_report=work/"output/validation/ctu_probability_distribution.csv",quality_report=work/"output/validation/data_quality_report.csv",feature_profile="all-cleaned"))
    predfile=work/"new_flows.csv"; pd.concat([benign.sample(10,random_state=4),bot.sample(10,random_state=5)]).to_csv(predfile,index=False); predict_command(argparse.Namespace(input=predfile,model=work/"output/models/c2_model.json",output=work/"output/reports/forensic_report.json",top_reasons=3,threshold=.5))
    print("\n  SELF-TEST PASSED"); return 0
