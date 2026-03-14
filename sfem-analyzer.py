#!/usr/bin/env python3
"""
Title: Salesforce ListView Event Log Analyser
Author: Eoghan Casey & Claude AI
Date: 03-07-2026
=========================================================
Parses Salesforce ListView event logs and detects:
  - Volume spikes       - daily rows-processed z-score vs. per-user baseline
  - Behavioral drift    - new objects, off-hours access
  - Enumeration bursts  - rapid sequential object sweeps
  - Developer name changes - scope escalation, view pivots

Usage:
    python salesforce_listview_analyzer_pandas.py --log events.csv [--baseline baseline.csv]
    python salesforce_listview_analyzer_pandas.py --log events.csv --output report.json
    python salesforce_listview_analyzer_pandas.py --demo
"""

import argparse
import json
import sys
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  Constants & thresholds
# ─────────────────────────────────────────────
SPIKE_ZSCORE_THRESHOLD = 2.5
MIN_BASELINE_DAYS      = 4
MIN_BASELINE_ROWS      = 50
BURST_WINDOW_SECONDS   = 60
BURST_MIN_REQUESTS     = 10
NEW_OBJECT_ALERT       = True
OFF_HOURS_START        = 22
OFF_HOURS_END          = 6
HIGH_ROW_THRESHOLD     = 300

SENSITIVE_OBJECTS: set = {
    "Account", "Contact", "Opportunity", "Lead", "Contract",
    "Case", "User", "ContentDocument", "Attachment", "Quote",
    "Order", "PricebookEntry", "PermissionSet", "Profile",
}

# CSV header → internal column name
COL_MAP = {
    "Date":               "timestamp",
    "Service":            "service",
    "Salesforce Org ID":  "org_id",
    "Event Name":         "event_type",
    "User Email":         "user_name",
    "User ID":            "user_id",
    "Queried Entities":   "entity_name",
    "Developer name":     "developer_name",
    "Event identifier":   "request_id",
    "Rows Processed":     "rows_processed",
    "Column headers":     "column_headers",
    "Number of columns":  "num_columns",
    "Content":            "content",
}

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def severity(score) -> str:
    s = 0.0 if pd.isna(score) else float(score)
    if s >= 8: return "CRITICAL"
    if s >= 5: return "HIGH"
    if s >= 3: return "MEDIUM"
    return "LOW"

def is_broad(name: str) -> bool:
    n = str(name).lower()
    return any(n.startswith(p) for p in ("all", "every", "org_wide", "global_"))

def is_scoped(name: str) -> bool:
    n = str(name).lower()
    return any(n.startswith(p) for p in ("my_", "my", "owned_", "team_", "queue_", "assigned_"))

def is_off_hours(hour: int) -> bool:
    h = int(hour)
    return h >= OFF_HOURS_START or h < OFF_HOURS_END


# ─────────────────────────────────────────────
#  Loading  ->  pd.DataFrame
# ─────────────────────────────────────────────
def load_events(source: str, from_string: bool = False) -> pd.DataFrame:
    """
    Read the CSV log, rename columns to internal names, add derived
    columns (date, hour, developer_name_lower).
    Returns a DataFrame sorted by timestamp.
    """
    df = pd.read_csv(StringIO(source) if from_string else source)
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    df["rows_processed"] = pd.to_numeric(
        df.get("rows_processed", 0), errors="coerce").fillna(0).astype(int)
    df["num_columns"] = pd.to_numeric(
        df.get("num_columns", 0), errors="coerce").fillna(0).astype(int)

    for col in ("user_name", "user_id", "org_id", "entity_name",
                "developer_name", "request_id", "column_headers",
                "content", "service", "event_type"):
        df[col] = df[col].fillna("").astype(str) if col in df.columns else ""

    # Fall back to user_id when email is absent
    df["user_name"] = df["user_name"].where(df["user_name"] != "", df["user_id"])

    # Lowercase dev name used for all comparisons
    df["developer_name_lower"] = df["developer_name"].str.lower()

    # Derived helper columns
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.hour

    return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────
#  Baseline builder  ->  dict of DataFrames / Series
# ─────────────────────────────────────────────
def build_baseline(df: pd.DataFrame) -> dict:
    """
    Compute per-user baseline statistics from historical events.

    Returns:
      row_stats          DataFrame  user_id | mean_rows | std_rows | baseline_days
      known_objects      Series     user_id  -> frozenset[entity_name]
      active_hours       Series     user_id  -> frozenset[hour]
      devnames_by_entity Series     (user_id, entity_name) -> frozenset[dev_name_lower]
    """
    df = df.head(MIN_BASELINE_ROWS)

    # Daily row totals per user, then aggregate to mean / std / day-count
    daily = (
        df.groupby(["user_id", "date"])["rows_processed"]
        .sum()
        .reset_index(name="daily_rows")
    )
    row_stats = (
        daily.groupby("user_id")["daily_rows"]
        .agg(mean_rows="mean", std_rows="std", baseline_days="count")
        .reset_index()
    )
    row_stats["std_rows"] = row_stats["std_rows"].fillna(0.0)

    # Per-user frozensets — O(1) membership tests during detection
    known_objects = df.groupby("user_id")["entity_name"].apply(frozenset)
    active_hours  = df.groupby("user_id")["hour"].apply(frozenset)
    devnames_by_entity = (
        df[df["developer_name_lower"] != ""]
        .groupby(["user_id", "entity_name"])["developer_name_lower"]
        .apply(frozenset)
    )

    return {
        "row_stats":          row_stats,
        "known_objects":      known_objects,
        "active_hours":       active_hours,
        "devnames_by_entity": devnames_by_entity,
    }


# ─────────────────────────────────────────────
#  Detection: volume / row spikes
# ─────────────────────────────────────────────
def detect_volume_spikes(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """
    Compute daily row totals per user, join with baseline mean/std,
    then flag rows where z-score >= SPIKE_ZSCORE_THRESHOLD.
    Falls back to a hard 1000 rows/day threshold when baseline is too thin.
    """
    daily = (
        df.groupby(["user_id", "date"])
        .agg(rows_processed=("rows_processed", "sum"),
             user_name=("user_name", "first"))
        .reset_index()
    )

    merged = daily.merge(baseline["row_stats"], on="user_id", how="left")
    merged["baseline_days"] = merged["baseline_days"].fillna(0).astype(int)
    merged["mean_rows"]     = merged["mean_rows"].fillna(0.0)
    merged["std_rows"]      = merged["std_rows"].fillna(0.0)

    # Thin-baseline path
    thin_mask = merged["baseline_days"] < MIN_BASELINE_DAYS
    thin = merged[thin_mask & (merged["rows_processed"] > 1000)].copy()
    thin["type"]       = "ROW_SPIKE_NO_BASELINE"
    thin["risk_score"] = (thin["rows_processed"] // 300).clip(upper=10)
    thin["severity"]   = thin["risk_score"].map(severity)
    thin["note"]       = "No historic baseline; total rows exceeds 1000/day"
    thin["timestamp"]  = thin["date"]

    # Z-score path
    thick = merged[~thin_mask].copy()
    thick["zscore"] = np.where(
        thick["std_rows"] > 0,
        (thick["rows_processed"] - thick["mean_rows"]) / thick["std_rows"],
        0.0,
    )
    hits = thick[thick["zscore"] >= SPIKE_ZSCORE_THRESHOLD].copy()
    hits["type"]           = "ROW_SPIKE"
    hits["risk_score"]     = (3 + hits["zscore"]).round().clip(upper=10).astype(int)
    hits["severity"]       = hits["risk_score"].map(severity)
    hits["baseline_mean"]  = hits["mean_rows"].round(2)
    hits["baseline_stdev"] = hits["std_rows"].round(2)
    hits["zscore"]         = hits["zscore"].round(2)
    hits["timestamp"]      = hits["date"]

    keep = ["type", "user_id", "user_name", "timestamp", "rows_processed",
            "risk_score", "severity", "baseline_mean", "baseline_stdev", "zscore", "note"]

    return pd.concat(
        [thin.reindex(columns=keep), hits.reindex(columns=keep)],
        ignore_index=True,
    )


# ─────────────────────────────────────────────
#  Detection: enumeration bursts
# ─────────────────────────────────────────────
def detect_enumeration_bursts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user forward sliding window (np.searchsorted on int64 timestamps).
    Flags windows of >= BURST_MIN_REQUESTS events within BURST_WINDOW_SECONDS.
    """
    records = []
    window_ns = int(BURST_WINDOW_SECONDS * 1e9)

    for uid, group in df.groupby("user_id"):
        grp   = group.sort_values("timestamp").reset_index(drop=True)
        ts_ns = grp["timestamp"].values.astype("int64")
        n, i  = len(grp), 0

        while i < n:
            j_end = int(np.searchsorted(ts_ns, ts_ns[i] + window_ns, side="right"))
            if j_end - i >= BURST_MIN_REQUESTS:
                burst     = grp.iloc[i:j_end]
                objects   = set(burst["entity_name"])
                sensitive = sorted(objects & SENSITIVE_OBJECTS)
                duration  = (ts_ns[j_end - 1] - ts_ns[i]) / 1e9
                risk      = min(10, 4 + len(objects))
                records.append({
                    "type":               "ENUMERATION_BURST",
                    "user_id":            uid,
                    "user_name":          burst["user_name"].iloc[0],
                    "timestamp":          str(burst["timestamp"].iloc[0]),
                    "burst_start":        str(burst["timestamp"].iloc[0]),
                    "burst_end":          str(burst["timestamp"].iloc[-1]),
                    "duration_seconds":   round(duration, 1),
                    "request_count":      len(burst),
                    "objects_accessed":   sorted(objects),
                    "sensitive_objects":  sensitive,
                    "total_rows_returned": int(burst["rows_processed"].sum()),
                    "risk_score":         risk,
                    "severity":           severity(risk),
                })
                i = j_end
            else:
                i += 1

    return pd.DataFrame(records) if records else pd.DataFrame()


# ─────────────────────────────────────────────
#  Detection: behavioral drift
# ─────────────────────────────────────────────
def detect_behavioral_drift(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """
    Vectorised checks (via Series.map + DataFrame.apply) for:
      NEW_USER_SENSITIVE_OBJECT  - no baseline, hits sensitive object
      NEW_OBJECT_ACCESS          - object not in user's baseline frozenset
      OFF_HOURS_ACCESS           - hour absent from user's active-hour frozenset
    """
    known_objects = baseline["known_objects"]
    active_hours  = baseline["active_hours"]

    work = df.copy()
    work["_ko"] = work["user_id"].map(known_objects)
    work["_ah"] = work["user_id"].map(active_hours)
    work["_has_bl"] = work["_ko"].apply(lambda x: isinstance(x, frozenset))

    parts = []

    # New user on sensitive object
    mask = ~work["_has_bl"] & work["entity_name"].isin(SENSITIVE_OBJECTS)
    chunk = work[mask].copy()
    chunk["type"]       = "NEW_USER_SENSITIVE_OBJECT"
    chunk["risk_score"] = 6
    chunk["severity"]   = "HIGH"
    chunk["note"]       = "User has no baseline history"
    parts.append(chunk)

    # New object for known user
    if NEW_OBJECT_ALERT:
        known = work[work["_has_bl"]].copy()
        nm = known.apply(lambda r: r["entity_name"] not in r["_ko"], axis=1)
        chunk = known[nm].copy()
        chunk["type"] = "NEW_OBJECT_ACCESS"
        chunk["risk_score"] = chunk["entity_name"].apply(
            lambda e: 7 if e in SENSITIVE_OBJECTS else 4)
        chunk["severity"] = chunk["risk_score"].map(severity)
        chunk["note"] = chunk["entity_name"].map(
            lambda e: f"{e} not in user's historic object scope")
        parts.append(chunk)

    # Off-hours access in an unrecognised hour
    known = work[work["_has_bl"]].copy()
    off_m = known["hour"].apply(is_off_hours)
    new_h = known.apply(lambda r: r["hour"] not in r["_ah"], axis=1)
    chunk = known[off_m & new_h].copy()
    chunk["type"]       = "OFF_HOURS_ACCESS"
    chunk["risk_score"] = 5
    chunk["severity"]   = "HIGH"
    chunk["hour_utc"]   = chunk["hour"]
    parts.append(chunk)

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return pd.DataFrame()

    out = pd.concat(non_empty, ignore_index=True)
    keep = ["type", "user_id", "user_name", "org_id", "timestamp",
            "entity_name", "developer_name", "rows_processed",
            "column_headers", "num_columns", "risk_score", "severity",
            "note", "hour_utc"]
    return _deduplicate(out.reindex(columns=keep))


# ─────────────────────────────────────────────
#  Detection: developer name changes
# ─────────────────────────────────────────────
def detect_developer_name_change(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """
    Per-row evaluation of three signals against per-(user, entity) frozensets:
      NEW_DEVELOPER_NAME    - name never seen for this user+entity combo
      SCOPE_ESCALATION      - was scoped (My_*); now accessing a broad (All*) view
      DEVELOPER_NAME_PIVOT  - abandoned all baseline names for the entity
    """
    devnames_by_entity = baseline["devnames_by_entity"]
    work = df[df["developer_name_lower"] != ""].copy()

    # Map frozensets onto each row via a MultiIndex lookup
    midx = pd.MultiIndex.from_arrays([work["user_id"], work["entity_name"]])
    work["_dns"] = midx.map(devnames_by_entity)
    work["_has_bl"]   = work["_dns"].apply(lambda x: isinstance(x, frozenset) and len(x) > 0)
    work["_is_broad"]  = work["developer_name_lower"].apply(is_broad)

    records = []

    for _, row in work.iterrows():
        dn    = row["developer_name_lower"]
        ent   = row["entity_name"]
        known = row["_dns"] if isinstance(row["_dns"], frozenset) else frozenset()

        base = dict(
            user_id   = row["user_id"],
            user_name = row["user_name"],
            org_id    = row["org_id"],
            timestamp = str(row["timestamp"]),
            entity_name  = ent,
            developer_name           = dn,
            baseline_developer_names = sorted(known),
            rows_processed  = row["rows_processed"],
            column_headers  = row["column_headers"],
            num_columns     = row["num_columns"],
        )

        if not row["_has_bl"]:
            if row["_is_broad"] and ent in SENSITIVE_OBJECTS:
                records.append({**base, "type": "NEW_DEVELOPER_NAME",
                    "risk_score": 6, "severity": "HIGH",
                    "note": "No baseline history; broad-scope developer name on sensitive object"})
            continue

        if dn not in known:
            risk = 7 if (ent in SENSITIVE_OBJECTS or row["_is_broad"]) else 4
            records.append({**base, "type": "NEW_DEVELOPER_NAME",
                "risk_score": risk, "severity": severity(risk),
                "note": f"Developer name '{dn}' not in baseline for {ent}"})

        if row["_is_broad"] and known and all(is_scoped(k) for k in known):
            risk = 8 if ent in SENSITIVE_OBJECTS else 5
            records.append({**base, "type": "SCOPE_ESCALATION",
                "risk_score": risk, "severity": severity(risk),
                "note": (f"User escalated from scoped views {sorted(known)} "
                         f"to broad view '{dn}' on {ent}")})

        if len(known) >= 2 and dn not in known and not (known & {dn}):
            risk = 5 if ent in SENSITIVE_OBJECTS else 3
            records.append({**base, "type": "DEVELOPER_NAME_PIVOT",
                "risk_score": risk, "severity": severity(risk),
                "note": (f"User completely abandoned usual dev names "
                         f"{sorted(known)} for '{dn}' on {ent}")})

    return _deduplicate(pd.DataFrame(records)) if records else pd.DataFrame()


# ─────────────────────────────────────────────
#  Detection: high row count
# ─────────────────────────────────────────────
def detect_high_row_access(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask — flag any row where rows_processed >= HIGH_ROW_THRESHOLD."""
    hits = df[df["rows_processed"] >= HIGH_ROW_THRESHOLD].copy()
    if hits.empty:
        return pd.DataFrame()
    hits["type"]       = "HIGH_ROW_COUNT"
    hits["risk_score"] = (4 + hits["rows_processed"] // 200).clip(upper=10).astype(int)
    hits["severity"]   = hits["risk_score"].map(severity)
    keep = ["type", "user_id", "user_name", "org_id", "timestamp",
            "entity_name", "developer_name", "rows_processed",
            "column_headers", "num_columns", "risk_score", "severity"]
    return hits.reindex(columns=keep).reset_index(drop=True)


# ─────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────
def _deduplicate(findings: pd.DataFrame) -> pd.DataFrame:
    if findings.empty:
        return findings
    subset = [c for c in ["type", "user_id", "timestamp", "entity_name"]
              if c in findings.columns]
    return findings.drop_duplicates(subset=subset).reset_index(drop=True)


# ─────────────────────────────────────────────
#  Report formatter
# ─────────────────────────────────────────────
def print_report(findings: pd.DataFrame, events: pd.DataFrame) -> None:
    SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    bar  = "=" * 72
    thin = "-" * 72

    print(f"\n{bar}")
    print("  SALESFORCE LISTVIEW ANOMALY REPORT")
    print(f"  Generated : {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Events    : {len(events)}")
    print(f"  Findings  : {len(findings)}")
    print(bar)

    if findings.empty:
        print("\n  No anomalies detected.\n")
        return

    sorted_f = (
        findings.copy()
        .assign(_o=findings.get("severity", pd.Series()).map(SEV_ORDER).fillna(3))
        .sort_values("_o")
        .drop(columns="_o")
        .reset_index(drop=True)
    )

    print("\n  SEVERITY SUMMARY")
    counts = sorted_f.get("severity", pd.Series()).value_counts()
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        n = int(counts.get(sev, 0))
        print(f"  {sev:<10} {n:>3}  {'*' * n}")

    print(f"\n{thin}\n  DETAILED FINDINGS\n{thin}")

    def g(row, key, default=""):
        v = row.get(key, default)
        return default if pd.isna(v) else str(v)

    for i, (_, f) in enumerate(sorted_f.iterrows(), 1):
        sev   = g(f, "severity", "LOW")
        ftype = g(f, "type", "UNKNOWN")
        uid   = g(f, "user_name") or g(f, "user_id")
        ts    = g(f, "timestamp") or g(f, "day")
        ent   = g(f, "entity_name")
        icons = {"CRITICAL": "[!!]", "HIGH": "[! ]", "MEDIUM": "[ !]", "LOW": "[  ]"}

        print(f"\n  [{i:02d}] {icons.get(sev,'[  ]')} {sev}  |  {ftype}")
        print(f"       User      : {uid}")
        if ts:  print(f"       Time      : {ts}")
        if ent: print(f"       Object    : {ent}")
        org = g(f, "org_id")
        if org: print(f"       Org ID    : {org}")

        if ftype in ("ROW_SPIKE", "ROW_SPIKE_NO_BASELINE"):
            print(f"       Rows/Day  : {g(f,'rows_processed')}  "
                  f"(baseline mean={g(f,'baseline_mean','?')} "
                  f"std={g(f,'baseline_stdev','?')}  "
                  f"z={g(f,'zscore','?')})")

        dn = g(f, "developer_name")
        if dn: print(f"       Dev Name  : {dn}")

        bdn = f.get("baseline_developer_names")
        if isinstance(bdn, list) and bdn:
            print(f"       Baseline  : {', '.join(bdn)}")

        if ftype not in ("ROW_SPIKE", "ROW_SPIKE_NO_BASELINE"):
            rp = g(f, "rows_processed")
            if rp: print(f"       Rows      : {rp}")

        ch, nc = g(f, "column_headers"), g(f, "num_columns")
        if ch: print(f"       Columns   : {ch}  ({nc} cols)")

        rc = g(f, "request_count")
        if rc: print(f"       Burst     : {rc} requests in {g(f,'duration_seconds','?')}s")

        so = f.get("sensitive_objects")
        if isinstance(so, list) and so:
            print(f"       Sensitive : {', '.join(so)}")

        note = g(f, "note")
        if note: print(f"       Note      : {note}")
        print(f"       Risk Score: {g(f,'risk_score','?')} / 10")

    print(f"\n{bar}\n")


# ─────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────
def analyze(
    log_source: str,
    baseline_source=None,
    from_string: bool = False,
) -> tuple:
    print("[*] Loading events...")
    events = load_events(log_source, from_string=from_string)
    print(f"[*] Loaded {len(events)} ListView events.")

    if baseline_source:
        print("[*] Loading external baseline...")
        baseline_df = load_events(baseline_source)
    else:
        split_idx   = max(1, int(len(events) * 0.8))
        baseline_df = events.iloc[:split_idx].copy()

    baseline = build_baseline(baseline_df)
    print(f"[*] Baseline built for {baseline['row_stats']['user_id'].nunique()} user(s).\n")

    parts = []

    print("[*] Running volume spike detection...")
    r = detect_volume_spikes(events, baseline)
    print(f"    -> {len(r)} spike(s) found");  parts.append(r)

    print("[*] Running enumeration burst detection...")
    r = detect_enumeration_bursts(events)
    print(f"    -> {len(r)} burst(s) found");  parts.append(r)

    print("[*] Running behavioral drift detection...")
    r = detect_behavioral_drift(events, baseline)
    print(f"    -> {len(r)} drift event(s) found");  parts.append(r)

    print("[*] Running developer name change detection...")
    r = detect_developer_name_change(events, baseline)
    print(f"    -> {len(r)} developer name change(s) found");  parts.append(r)

    print("[*] Running high row-count detection...")
    r = detect_high_row_access(events)
    print(f"    -> {len(r)} high-row event(s) found");  parts.append(r)

    findings = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return findings, events


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salesforce ListView anomaly detector (pandas edition)")
    parser.add_argument("--log",      help="Path to ListView event CSV log")
    parser.add_argument("--baseline", help="Optional separate baseline CSV")
    parser.add_argument("--output",   help="Write JSON findings to this file")
    parser.add_argument("--demo",     action="store_true",
                        help="Run against built-in synthetic demo data")
    args = parser.parse_args()

    if args.demo:
        print("[*] Running in DEMO mode with synthetic data.\n")
        findings, events = analyze(DEMO_CSV, from_string=True)
    elif args.log:
        findings, events = analyze(args.log, baseline_source=args.baseline)
    else:
        parser.print_help(); sys.exit(1)

    print_report(findings, events)

    if args.output:
        records = findings.where(findings.notna(), None).to_dict(orient="records")
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
        print(f"[*] JSON findings written to {args.output}\n")


if __name__ == "__main__":
    main()
