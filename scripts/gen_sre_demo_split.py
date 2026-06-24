#!/usr/bin/env python3
"""
Two SRE Q&A training splits for the appliance demo recording:
  data/sre-tables-train/sre_qa_v1.jsonl  — broad SRE Q&A, LIGHT on the OOM hero topic
  data/sre-tables-train/sre_qa_v2.jsonl  — v1 + deep OOM-remediation rows (the "appended"
                                            set) so the retrained answer is visibly richer

Format: {question, reference_answer, context}. Deterministic.
"""
import json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "sre-tables-train")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)

# broad use-case answers (short) — same shape as gen_sre_qa
BROAD = {
 "CPU Throttling": "CFS throttling from a tight CPU limit. Raise/remove the CPU limit; alert on throttled-periods ratio.",
 "Pod CrashLoop Prediction": "Repeated restarts from a failing dependency or ungraceful shutdown. Check logs --previous, run >=2 replicas, add a PDB.",
 "Disk Pressure": "Node ephemeral-storage filling. Move logs to stdout, set storage limits, alert before the eviction threshold.",
 "Node NotReady Risk": "Kubelet/PLEG unhealthy under load. Cordon+drain, spread replicas across nodes/zones.",
 "Network Latency Spike": "p99 latency / packet drops from a network or upstream regression. Add retries+timeouts, fail over the bad AZ.",
 "DNS Resolution Failures": "CoreDNS degradation. Run >=2 CoreDNS replicas + node-local cache, alert on DNS error rate.",
 "HTTP Error Surge": "5xx spike from a bad deploy or dependency. Roll back, gate rollouts on error-rate SLOs.",
 "Replica Starvation": "Ready replicas below desired. Scale up, fix readiness, set HPA with headroom.",
}
SVCS = ["payments-api","orders-api","checkout-web","auth-gateway","search-api","inventory-svc","ledger-svc"]
NS = {"payments-api":"payments","orders-api":"orders","checkout-web":"checkout","auth-gateway":"identity",
      "search-api":"search","inventory-svc":"warehouse","ledger-svc":"finance"}

def broad_rows(n):
    rows, ucs = [], list(BROAD.items())
    for i in range(n):
        uc, ans = ucs[i % len(ucs)]
        svc = SVCS[i % len(SVCS)]
        rows.append({"question": f"{svc} in {NS[svc]} was flagged for '{uc}'. What's the cause and fix?",
                     "reference_answer": f"Root cause + remediation: {ans}",
                     "context": f"service={svc} namespace={NS[svc]} use_case={uc}"})
    return rows

# Light OOM coverage for v1 (generic, short)
OOM_LIGHT = [{"question": "payments-api was flagged for OOM Risk. What do we do?",
              "reference_answer": "It's approaching its memory limit. Raise the memory limit and restart.",
              "context": "use_case=OOM Risk Forecast service=payments-api"}]

# Deep OOM remediation for v2 (the hero improvement)
OOM_DEEP_ANSWER = (
 "Root cause: the container's working set is approaching its memory limit; an OOMKill "
 "(exit 137) is imminent and will restart the pod. Remediation: (1) capture a heap "
 "profile to confirm leak vs. undersized limit; (2) if it's a recent regression, roll "
 "back the last deploy; (3) raise limits.memory above the observed p95 peak with "
 "headroom and set requests.memory to steady-state; (4) add a startupProbe so a slow "
 "boot isn't counted as a crash. Preventive measures: size requests/limits from real "
 "p95+buffer, alert at 80% of the memory limit, size DB/HTTP connection pools and add "
 "retry-with-backoff so a restart storm doesn't exhaust them, and load-test before "
 "release so the ceiling is known, not discovered in production.")
def oom_deep(n):
    forms = [
      "Our {svc} pod in {ns} was flagged for OOM Risk at high risk. Give the full root cause and remediation.",
      "{svc} keeps approaching its memory limit (OOM Risk). What is the complete remediation and prevention?",
      "How should an SRE handle an OOM Risk prediction on {svc}? Root cause + full fix please.",
      "{svc} in {ns}: OOMKilled risk is high. Walk me through diagnosis, remediation, and prevention.",
    ]
    rows = []
    for i in range(n):
        svc = SVCS[i % len(SVCS)]
        rows.append({"question": forms[i % len(forms)].format(svc=svc, ns=NS[svc]),
                     "reference_answer": OOM_DEEP_ANSWER,
                     "context": f"use_case=OOM Risk Forecast service={svc} namespace={NS[svc]} risk=high"})
    return rows

v1 = broad_rows(140) + OOM_LIGHT
rng.shuffle(v1)
v2 = list(v1) + oom_deep(100)   # appended deep OOM rows
rng.shuffle(v2)

for name, rows in [("sre_qa_v1.jsonl", v1), ("sre_qa_v2.jsonl", v2)]:
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {p}")
