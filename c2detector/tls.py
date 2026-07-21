from __future__ import annotations
from pathlib import Path
from collections import Counter
import pandas as pd


def extract_tls_metadata(pcap_path: Path) -> tuple[pd.DataFrame, dict]:
    """Extract TLS metadata for forensic enrichment only.

    This output is not fed into the CICFlowMeter model unless a separate TLS model is trained.
    """
    try:
        import pyshark
    except ImportError as exc:
        raise RuntimeError("TLS enrichment requires pyshark. Install it with: pip install pyshark") from exc
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    coverage = Counter(); records = []
    cap = pyshark.FileCapture(str(pcap_path), keep_packets=False)
    try:
        for pkt in cap:
            coverage["total_packets"] += 1
            highest = str(getattr(pkt, "highest_layer", "UNKNOWN")).upper()
            coverage[f"highest_{highest}"] += 1
            if hasattr(pkt, "tcp"): coverage["tcp_packets"] += 1
            if hasattr(pkt, "udp"): coverage["udp_packets"] += 1
            if hasattr(pkt, "dns"): coverage["dns_packets"] += 1
            if hasattr(pkt, "http"): coverage["http_packets"] += 1
            if not hasattr(pkt, "tls"):
                continue
            coverage["tls_packets"] += 1
            ip_layer = getattr(pkt, "ip", None) or getattr(pkt, "ipv6", None)
            transport = getattr(pkt, "tcp", None) or getattr(pkt, "udp", None)
            if ip_layer is None or transport is None:
                continue
            records.append({
                "timestamp": float(getattr(pkt, "sniff_timestamp", 0.0)),
                "src_ip": str(getattr(ip_layer, "src", "unknown")),
                "dst_ip": str(getattr(ip_layer, "dst", "unknown")),
                "src_port": int(getattr(transport, "srcport", 0)),
                "dst_port": int(getattr(transport, "dstport", 0)),
                "protocol": "TCP" if hasattr(pkt, "tcp") else "UDP",
                "ja3": str(getattr(pkt.tls, "handshake_ja3", "unknown")),
                "ja3s": str(getattr(pkt.tls, "handshake_ja3s", "unknown")),
                "cipher_suite": str(getattr(pkt.tls, "handshake_ciphersuite", "unknown")),
                "sni": str(getattr(pkt.tls, "handshake_extensions_server_name", "unknown")),
                "tls_version": str(getattr(pkt.tls, "record_version", "unknown")),
            })
    finally:
        cap.close()
    coverage["tls_records_extracted"] = len(records)
    return pd.DataFrame(records), dict(coverage)


def enrich_alerts_with_tls(alerts: list[dict], tls_df: pd.DataFrame) -> dict:
    if tls_df.empty:
        return {"matched_alerts": 0, "tls_records": 0}
    matched = 0
    for alert in alerts:
        ids = alert.get("network_identifiers", {})
        src = str(ids.get("Src IP", ids.get("Source IP", "")))
        dst = str(ids.get("Dst IP", ids.get("Destination IP", "")))
        src_port = ids.get("Src Port", ids.get("Source Port"))
        dst_port = ids.get("Destination Port", ids.get("Dst Port"))
        mask = pd.Series(True, index=tls_df.index)
        if src: mask &= tls_df["src_ip"].astype(str).eq(src)
        if dst: mask &= tls_df["dst_ip"].astype(str).eq(dst)
        if src_port is not None: mask &= tls_df["src_port"].astype(str).eq(str(src_port))
        if dst_port is not None: mask &= tls_df["dst_port"].astype(str).eq(str(dst_port))
        matches = tls_df.loc[mask]
        if matches.empty:
            continue
        row = matches.iloc[0]
        alert["tls_forensic_context"] = {
            "ja3": row.get("ja3", "unknown"), "ja3s": row.get("ja3s", "unknown"),
            "cipher_suite": row.get("cipher_suite", "unknown"), "sni": row.get("sni", "unknown"),
            "tls_version": row.get("tls_version", "unknown"),
        }
        matched += 1
    return {"matched_alerts": matched, "tls_records": len(tls_df)}
