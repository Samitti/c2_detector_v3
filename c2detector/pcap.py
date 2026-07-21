from __future__ import annotations
from pathlib import Path
import logging, shutil, subprocess
from .data import read_csv_checked
from .utils import ensure_parent

def run_cicflowmeter(pcap_path:Path,csv_output:Path,executable:str,*,overwrite:bool):
    if not pcap_path.exists(): raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    if pcap_path.suffix.lower() not in {".pcap",".pcapng",".cap"}: raise ValueError(f"Unsupported capture extension '{pcap_path.suffix}'.")
    resolved=shutil.which(executable)
    if resolved is None: raise RuntimeError(f"CICFlowMeter executable '{executable}' was not found. Activate your virtual environment or pass --cicflowmeter-bin.")
    ensure_parent(csv_output)
    if csv_output.exists():
        if not overwrite: raise ValueError(f"Flow CSV already exists: {csv_output}. Use --overwrite-flow-csv.")
        csv_output.unlink()
    command=[resolved,"-f",str(pcap_path),"-c",str(csv_output)]; print("  Flow extractor command : "+" ".join(command)); logging.info("Running flow extractor: %s",command)
    cp=subprocess.run(command,check=False,capture_output=True,text=True)
    if cp.returncode!=0: raise RuntimeError(f"CICFlowMeter failed with exit code {cp.returncode}: {cp.stderr.strip() or cp.stdout.strip() or 'no diagnostic output'}")
    if not csv_output.exists() or csv_output.stat().st_size==0: raise RuntimeError(f"CICFlowMeter did not create a usable CSV: {csv_output}")
    extracted=read_csv_checked(csv_output)
    if extracted.empty: raise RuntimeError("CICFlowMeter produced an empty CSV.")
    return {"name":"CICFlowMeter CLI","executable":resolved,"command":command,"return_code":cp.returncode,"flow_csv":str(csv_output),"flows_extracted":len(extracted)}
