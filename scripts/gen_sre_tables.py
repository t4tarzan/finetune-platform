#!/usr/bin/env python3
"""
Synthetic data generator for the Predictive-SRE / Hermes platform (38 tables).

Design: build ONE coherent world (clusters → namespaces → nodes → pods, agents,
ml_use_cases, SOPs) over a 30-day window, assign each pod a failure scenario, then
derive every table from it so keys (pod/namespace/cluster/node) and timestamps stay
consistent and failure storylines correlate across tables (metrics → events →
predictions → alerts → feedback). Output: one CSV per table under ./out/.

JSONB columns are emitted as JSON strings; array columns as JSON arrays. That loads
cleanly via the platform's CSV→DuckDB importer. (For Postgres COPY, arrays would
need {..} literals — see README.)

Deterministic: fixed SEED. Stdlib only.

Usage:  python3 generate.py
"""
import csv, json, os, random, uuid, hashlib
from datetime import datetime, timedelta

SEED = 42
rng = random.Random(SEED)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sre-tables")
os.makedirs(OUT, exist_ok=True)

NOW = datetime(2026, 6, 23, 12, 0, 0)
START = NOW - timedelta(days=30)

def U(): return str(uuid.UUID(int=rng.getrandbits(128)))
def ts(dt): return dt.strftime("%Y-%m-%d %H:%M:%S")
def between(a, b): return a + (b - a) * rng.random()
def tbetween(a, b): return a + timedelta(seconds=rng.random() * (b - a).total_seconds())
def pick(seq): return rng.choice(seq)
def chance(p): return rng.random() < p
def jj(o): return json.dumps(o, separators=(",", ":"))

def write(table, header, rows):
    path = os.path.join(OUT, f"{table}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for r in rows:
            wr.writerow(["" if v is None else (jj(v) if isinstance(v, (dict, list)) else
                         ("true" if v is True else "false" if v is False else v)) for v in r])
    print(f"  {table:30} {len(rows):>6} rows")
    return len(rows)

# ─────────────────────────────────────────────────────────────────────────────
#  WORLD
# ─────────────────────────────────────────────────────────────────────────────
CLUSTERS = [("prod-use2-eks", "prod"), ("prod-usw2-eks", "prod"),
            ("staging-use2-eks", "staging"), ("dr-use1-eks", "dr")]
ENV_OF = dict(CLUSTERS)

NAMESPACES = ["payments", "orders", "identity", "search", "inventory", "checkout",
              "notifications", "ledger", "platform", "observability", "kube-system"]

# service base name -> (namespace, kind, base_cpu_m, base_mem_mi, replicas)
SERVICES = [
    ("payments-api", "payments", 500, 1024, 4), ("payments-worker", "payments", 300, 768, 3),
    ("orders-api", "orders", 400, 768, 3), ("orders-worker", "orders", 300, 512, 3),
    ("auth-gateway", "identity", 500, 512, 4), ("session-store", "identity", 250, 1536, 2),
    ("search-api", "search", 600, 2048, 4), ("indexer", "search", 800, 3072, 2),
    ("inventory-svc", "inventory", 300, 512, 3), ("checkout-web", "checkout", 400, 512, 4),
    ("cart-svc", "checkout", 300, 384, 3), ("notifications", "notifications", 200, 384, 2),
    ("ledger-svc", "ledger", 500, 1024, 3), ("recommendation", "search", 700, 2560, 2),
    ("image-resizer", "platform", 300, 512, 2), ("config-svc", "platform", 150, 256, 2),
    ("coredns", "kube-system", 100, 170, 2), ("metrics-server", "kube-system", 100, 256, 1),
]

SCENARIOS = ["healthy", "healthy", "healthy", "healthy", "memory_leak", "cpu_throttle",
             "oom_crashloop", "disk_pressure", "network_latency", "node_pressure"]

def build_world():
    nodes = {}     # cluster -> [node dicts]
    for c, env in CLUSTERS:
        n = 6 if env == "prod" else 3
        nodes[c] = [{
            "name": f"ip-10-{rng.randint(0,40)}-{rng.randint(0,255)}-{rng.randint(0,255)}.{c.split('-')[1]}.compute.internal",
            "cluster": c, "ready": chance(0.95),
            "cpu_cap": pick([8, 16, 16, 32]), "mem_cap_gi": pick([32, 64, 64, 128]),
            "pod_cap": pick([58, 110, 110]),
        } for _ in range(n)]

    pods = []
    for c, env in CLUSTERS:
        # staging/dr run a subset
        svcs = SERVICES if env == "prod" else rng.sample(SERVICES, k=int(len(SERVICES) * 0.6))
        for (svc, nsd, cpu, mem, rep) in svcs:
            reps = rep if env == "prod" else max(1, rep // 2)
            scenario = pick(SCENARIOS)
            onset = tbetween(START + timedelta(days=2), NOW - timedelta(hours=6)) if scenario != "healthy" else None
            for i in range(reps):
                node = pick(nodes[c])
                h = hashlib.md5(f"{c}{svc}{i}{SEED}".encode()).hexdigest()[:10]
                pods.append({
                    "pod": f"{svc}-{h[:5]}{rng.randint(10,99)}-{h[5:]}",
                    "svc": svc, "ns": nsd, "cluster": c, "env": env, "node": node["name"],
                    "cpu_req": cpu, "cpu_lim": int(cpu * pick([1.5, 2, 2])),
                    "mem_req": mem, "mem_lim": int(mem * pick([1.3, 1.5, 2])),
                    "scenario": scenario if i == 0 or chance(0.6) else "healthy",
                    "onset": onset,
                    "created": tbetween(START - timedelta(days=20), START + timedelta(days=5)),
                    "restarts": 0,
                })
    # use cases
    use_cases = [
        (1, "Memory Leak Detection", "P1", "memory", "Predict container memory exhaustion before OOMKill"),
        (2, "OOM Risk Forecast", "P1", "memory", "Forecast out-of-memory kill probability per pod"),
        (3, "CPU Throttling", "P2", "cpu", "Detect sustained CFS throttling impacting latency"),
        (4, "Pod CrashLoop Prediction", "P1", "availability", "Predict CrashLoopBackOff from restart trends"),
        (5, "Disk Pressure", "P2", "storage", "Predict node ephemeral-storage eviction"),
        (6, "Node NotReady Risk", "P1", "node", "Forecast node kubelet/PLEG failures"),
        (7, "Network Latency Spike", "P2", "network", "Detect p99 latency / packet-drop regressions"),
        (8, "DNS Resolution Failures", "P2", "network", "Predict CoreDNS degradation"),
        (9, "Replica Starvation", "P2", "availability", "Detect ready-replica shortfalls vs desired"),
        (10, "GC Pause Storms", "P3", "memory", "JVM GC pause clustering / heap pressure"),
        (11, "Cert Expiry", "P2", "security", "Upcoming TLS certificate expiry"),
        (12, "Connection Pool Exhaustion", "P1", "availability", "DB/HTTP connection pool saturation"),
        (13, "HTTP Error Surge", "P2", "availability", "5xx error-rate anomaly"),
        (14, "Cluster Capacity", "P2", "node", "Cluster CPU/mem capacity headroom"),
        (15, "Security Policy Violation", "P3", "security", "Unauthorized access / policy breaches"),
    ]
    # agents (multi-agent fleet)
    layers = [("worker", "worker", 24), ("cluster_manager", "manager", 8), ("strategist", "strategist", 4)]
    agents = []
    for c, env in CLUSTERS:
        for (atype, layer, _) in layers:
            cnt = {"worker": 6, "cluster_manager": 2, "strategist": 1}[atype]
            for k in range(cnt):
                agents.append({
                    "agent_id": f"{atype}-{c}-{k:02d}", "type": atype, "layer": layer,
                    "cluster": c, "env": env,
                })
    # SOPs
    sop_domains = ["memory", "cpu", "availability", "network", "storage", "node", "security"]
    sops = []
    for i in range(40):
        dom = pick(sop_domains)
        sops.append({
            "id": i + 1, "domain": dom,
            "name": f"{dom.title()} Runbook {i+1:02d}: " + pick([
                "OOMKilled remediation", "CrashLoopBackOff triage", "CFS throttling tuning",
                "Node drain & cordon", "DiskPressure eviction recovery", "CoreDNS restart",
                "PDB & graceful shutdown", "Connection pool sizing", "Cert rotation",
                "Readiness probe tuning", "HPA scaling response", "Memory leak rollback"]),
            "path": f"sops/{dom}/runbook-{i+1:02d}.md",
        })
    return {"nodes": nodes, "pods": pods, "use_cases": use_cases, "agents": agents, "sops": sops}

W = build_world()
PODS, AGENTS, USE_CASES, SOPS = W["pods"], W["agents"], W["use_cases"], W["sops"]
ALL_NODES = [n for ns in W["nodes"].values() for n in ns]
print(f"world: {len(PODS)} pods, {len(ALL_NODES)} nodes, {len(AGENTS)} agents, "
      f"{len(USE_CASES)} use_cases, {len(SOPS)} sops")

# scenario → metric shaping helpers
def scen_metrics(p, t):
    """Return dict of scenario-shaped metric multipliers at time t for pod p."""
    sev = 0.0
    if p["onset"] and t >= p["onset"]:
        sev = min(1.0, (t - p["onset"]).total_seconds() / (3 * 86400))  # ramps over ~3d
    s = p["scenario"]
    m = dict(mem=between(0.35, 0.6), cpu=between(0.2, 0.5), restarts=0, oom=0, ready=True,
             rt=between(40, 120), err=between(0, 0.01), drops=0, throttle=0, io=between(2, 20))
    if s == "memory_leak":
        m["mem"] = min(0.99, 0.5 + 0.5 * sev)
        if sev > 0.8: m["oom"], m["restarts"], m["ready"] = 1, rng.randint(1, 4), chance(0.5)
    elif s == "oom_crashloop":
        m["mem"] = min(0.99, 0.7 + 0.3 * sev); m["restarts"] = int(2 + 12 * sev)
        m["oom"] = int(1 + 6 * sev); m["ready"] = sev < 0.5
    elif s == "cpu_throttle":
        m["cpu"] = min(0.99, 0.6 + 0.4 * sev); m["throttle"] = between(0.2, 0.7) * sev; m["rt"] = between(120, 600) * (1 + sev)
    elif s == "disk_pressure":
        m["io"] = between(40, 200) * (1 + sev)
        if sev > 0.7: m["restarts"], m["ready"] = rng.randint(0, 2), chance(0.7)
    elif s == "network_latency":
        m["rt"] = between(200, 900) * (1 + sev); m["drops"] = int(50 * sev); m["err"] = between(0.01, 0.08) * (1 + sev)
    elif s == "node_pressure":
        m["mem"] = min(0.97, 0.5 + 0.4 * sev); m["cpu"] = min(0.97, 0.5 + 0.3 * sev); m["ready"] = sev < 0.6
    return m, sev

# A pool of pods that have active incidents (for predictions/alerts/events)
INCIDENT_PODS = [p for p in PODS if p["scenario"] != "healthy"]

SUMMARY = {}
def emit(table, header, rows): SUMMARY[table] = write(table, header, rows)

# convenience time series of timestamps across the window
def series(n, a=START, b=NOW):
    step = (b - a) / n
    return [a + step * i + timedelta(seconds=rng.randint(-30, 30)) for i in range(n)]

ENVIRONMENTS = ["prod", "staging", "dr"]

# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 1 — Kubernetes observability
# ═══════════════════════════════════════════════════════════════════════════

# kubernetes_pod_status — snapshots over time
def g_kubernetes_pod_status():
    hdr = ["id","cluster_name","namespace","pod_name","node_name","phase","ready_status",
           "restart_count","cpu_request","cpu_limit","memory_request","memory_limit",
           "source_environment","sample_rate","collected_at"]
    rows = []; i = 0
    for t in series(70):  # 70 collection cycles
        for p in PODS:
            m, sev = scen_metrics(p, t)
            phase = "Running"
            if p["scenario"] == "oom_crashloop" and sev > 0.5: phase = pick(["CrashLoopBackOff","Running","Error"])
            elif not m["ready"]: phase = pick(["Running","Pending"])
            i += 1
            rows.append([i, p["cluster"], p["ns"], p["pod"], p["node"], phase, m["ready"],
                         m["restarts"], f"{p['cpu_req']}m", f"{p['cpu_lim']}m",
                         f"{p['mem_req']}Mi", f"{p['mem_lim']}Mi", p["env"],
                         round(between(0.1, 1.0), 2), ts(t)])
            if len(rows) >= 4000: break
        if len(rows) >= 4000: break
    emit("kubernetes_pod_status", hdr, rows)

def g_kubernetes_node_status():
    hdr = ["id","cluster_name","node_name","ready_status","cpu_capacity","memory_capacity",
           "pod_capacity","cpu_allocatable","memory_allocatable","pod_allocatable","collected_at"]
    rows = []; i = 0
    for t in series(80):
        for n in ALL_NODES:
            i += 1
            rows.append([i, n["cluster"], n["name"], n["ready"] and chance(0.97),
                         f"{n['cpu_cap']}", f"{n['mem_cap_gi']}Gi", n["pod_cap"],
                         f"{int(n['cpu_cap']*1000*0.95)}m", f"{int(n['mem_cap_gi']*0.92)}Gi",
                         n["pod_cap"]-rng.randint(2,8), ts(t)])
    emit("kubernetes_node_status", hdr, rows)

EVENT_TEMPLATES = {
    "oom_crashloop": [("Warning","BackOff","Back-off restarting failed container {svc} in pod {pod}"),
                      ("Warning","OOMKilling","Memory cgroup out of memory: Killed process in {pod}"),
                      ("Warning","Unhealthy","Liveness probe failed: HTTP 500 for {pod}")],
    "memory_leak": [("Warning","OOMKilling","Container {svc} exceeded memory limit; OOMKilled"),
                    ("Warning","Evicted","Pod {pod} evicted: node memory pressure")],
    "cpu_throttle": [("Warning","Unhealthy","Readiness probe failed: timeout for {pod}"),
                     ("Normal","Throttling","CPU throttled for container {svc}")],
    "disk_pressure": [("Warning","Evicted","Pod {pod} evicted: node was low on resource ephemeral-storage"),
                      ("Warning","FreeDiskSpaceFailed","Failed to garbage collect on node {node}")],
    "network_latency": [("Warning","Unhealthy","Readiness probe errored: dial tcp i/o timeout for {pod}"),
                        ("Warning","DNSConfigForming","Search line limits exceeded for {pod}")],
    "node_pressure": [("Warning","NodeNotReady","Node {node} status is now: NodeNotReady"),
                      ("Warning","Rebooted","Node {node} has been rebooted")],
    "healthy": [("Normal","Scheduled","Successfully assigned {pod} to {node}"),
                ("Normal","Pulled","Container image pulled for {svc}"),
                ("Normal","Started","Started container {svc}")],
}
def g_kubernetes_events():
    hdr = ["id","cluster_name","namespace","event_type","reason","message",
           "involved_object_kind","involved_object_name","source_component",
           "first_timestamp","last_timestamp","count","collected_at"]
    rows = []; i = 0
    for _ in range(1600):
        p = pick(PODS if chance(0.4) else (INCIDENT_PODS or PODS))
        etype, reason, msg = pick(EVENT_TEMPLATES.get(p["scenario"], EVENT_TEMPLATES["healthy"]))
        ft = tbetween(p["onset"] or START, NOW)
        cnt = rng.randint(1, 40) if etype == "Warning" else 1
        i += 1
        rows.append([i, p["cluster"], p["ns"], etype, reason,
                     msg.format(svc=p["svc"], pod=p["pod"], node=p["node"].split(".")[0]),
                     "Pod", p["pod"], pick(["kubelet","default-scheduler","controllermanager"]),
                     ts(ft), ts(ft + timedelta(minutes=rng.randint(0, 120))), cnt, ts(NOW)])
    emit("kubernetes_events", hdr, rows)

def g_comprehensive_metrics():
    hdr = ["id","pod_name","namespace","cluster_name","node_name","environment","jvm_heap_used",
           "jvm_heap_max","jvm_metaspace_used","container_memory_usage","container_memory_limit",
           "container_memory_rss","container_memory_working_set","memory_pressure","swap_usage",
           "cpu_usage_percent","cpu_throttled_periods","gc_duration_seconds","gc_count","thread_count",
           "thread_blocked_count","response_time_ms","network_latency_ms","packet_drops","tcp_connections",
           "dns_resolution_time_ms","http_request_count","http_error_rate","fs_usage_bytes","fs_limit_bytes",
           "io_wait_time_ms","disk_read_bytes","disk_write_bytes","pod_ready","restart_count","oom_events",
           "health_check_failures","startup_time_seconds","security_violations","unauthorized_access_attempts",
           "policy_violations","node_ready","node_memory_available","node_cpu_available",
           "cluster_capacity_percent","collection_timestamp","created_at"]
    rows = []; i = 0
    MiB = 1024*1024
    for t in series(60):
        for p in PODS:
            m, sev = scen_metrics(p, t)
            lim = p["mem_lim"]*MiB
            use = int(lim*m["mem"])
            i += 1
            rows.append([i, p["pod"], p["ns"], p["cluster"], p["node"], p["env"],
                int(use*0.7), int(lim*0.8), int(80*MiB), use, lim, int(use*0.85), int(use*0.95),
                round(m["mem"],4), 0, round(m["cpu"]*100,2), int(m["throttle"]*1000),
                round(between(0.001,0.05)*(1+sev),6), rng.randint(5,80), rng.randint(20,200),
                int(m.get("throttle",0)*30), round(m["rt"],2), round(m["rt"]*0.4,2), m["drops"],
                rng.randint(20,400), round(between(1,m["rt"]*0.2+5),2), rng.randint(1000,90000),
                round(m["err"],4), int(between(1,40)*MiB), int(60*MiB), round(m["io"],2),
                rng.randint(0,5_000_000), rng.randint(0,9_000_000), m["ready"], m["restarts"], m["oom"],
                rng.randint(0,3) if not m["ready"] else 0, round(between(2,30),2),
                0, 0, 0, chance(0.96), rng.randint(2,30)*1024*MiB, round(between(5,80),2),
                round(between(40,92),2), ts(t), ts(t)])
            if len(rows) >= 3500: break
        if len(rows) >= 3500: break
    emit("comprehensive_metrics", hdr, rows)

METRIC_NAMES = ["container_memory_working_set_bytes","container_cpu_usage_seconds_total",
    "container_memory_rss","kube_pod_container_status_restarts_total","http_request_duration_ms",
    "http_requests_total","container_fs_usage_bytes","container_network_receive_bytes_total",
    "go_gc_duration_seconds","process_open_fds","tcp_established"]
def g_metrics_realtime():
    hdr = ["id","metric_name","metric_value","unit","cluster_name","namespace","pod_name",
           "container_name","timestamp","metadata","node_name","source"]
    rows = []; i = 0
    for t in series(40):
        for p in rng.sample(PODS, k=min(len(PODS), 60)):
            mn = pick(METRIC_NAMES)
            i += 1
            rows.append([i, mn, round(between(0,1e6),3), pick(["bytes","cores","ms","count"]),
                         p["cluster"], p["ns"], p["pod"], p["svc"], ts(t),
                         {"unit_hint":"prom","scrape":"30s"}, p["node"], pick(["sysdig","prometheus"])])
            if len(rows) >= 2500: break
        if len(rows) >= 2500: break
    emit("metrics_realtime", hdr, rows)

def g_metrics_compressed():
    hdr = ["id","time_bucket","cluster_name","namespace","pod_name","metric_name","avg_value",
           "min_value","max_value","stddev_value","sample_count","percentile_95","percentile_99","created_at"]
    rows = []; i = 0
    for t in series(48):  # hourly-ish buckets
        for p in rng.sample(PODS, k=min(len(PODS), 30)):
            mn = pick(METRIC_NAMES); avg = between(10, 5000)
            i += 1
            rows.append([i, ts(t.replace(minute=0, second=0)), p["cluster"], p["ns"], p["pod"], mn,
                         round(avg,3), round(avg*0.6,3), round(avg*1.8,3), round(avg*0.15,3),
                         rng.randint(30,120), round(avg*1.5,3), round(avg*1.75,3), ts(t)])
            if len(rows) >= 1400: break
        if len(rows) >= 1400: break
    emit("metrics_compressed", hdr, rows)

def g_sysdig_metrics():
    hdr = ["id","timestamp","query_name","metric_name","metric_value","labels","cluster_name",
           "namespace","pod_name","node_name","ml_use_cases","collection_metadata","created_at","source"]
    rows = []; i = 0
    for t in series(45):
        for p in rng.sample(PODS, k=min(len(PODS), 55)):
            m, sev = scen_metrics(p, t)
            i += 1
            rows.append([i, ts(t), pick(["mem_usage","cpu_usage","restart_rate","http_5xx","net_latency"]),
                         pick(METRIC_NAMES), round(between(0,1e5),3),
                         {"pod":p["pod"],"ns":p["ns"],"scenario":p["scenario"]},
                         p["cluster"], p["ns"], p["pod"], p["node"],
                         rng.sample([1,2,3,4,7,13], k=rng.randint(1,3)),
                         {"collector":"sysdig","interval":30}, ts(t), "sysdig"])
            if len(rows) >= 2500: break
        if len(rows) >= 2500: break
    emit("sysdig_metrics", hdr, rows)

def g_ml_features():
    hdr = ["id","timestamp","container_memory_used","container_memory_limit","memory_utilization_ratio",
           "memory_growth_rate","container_cpu_used","container_cpu_limit","cpu_utilization_ratio",
           "cpu_throttled_periods","cpu_throttling_ratio","pod_ready_count","pod_total_count","pod_ready_ratio",
           "node_memory_capacity","node_cpu_capacity","node_memory_pressure","services_up_count",
           "services_total_count","service_availability_ratio","cluster_name","namespace","collection_source","created_at"]
    rows = []; i = 0
    MiB = 1024*1024
    for t in series(55):
        for p in rng.sample(PODS, k=min(len(PODS), 45)):
            m, sev = scen_metrics(p, t)
            tot = rng.randint(2,5); ready = tot if m["ready"] else rng.randint(0, tot-1)
            stot = rng.randint(8,18); sup = stot - (0 if m["ready"] else rng.randint(1,3))
            i += 1
            rows.append([i, ts(t), int(p["mem_lim"]*MiB*m["mem"]), p["mem_lim"]*MiB, round(m["mem"],4),
                         round(between(-0.01,0.05)*(1+sev),6), round(m["cpu"]*p["cpu_lim"]/1000,4),
                         p["cpu_lim"]/1000, round(m["cpu"],4), int(m["throttle"]*1000), round(m["throttle"],4),
                         ready, tot, round(ready/tot,4), 64*1024*MiB, 16.0, sev>0.7 and chance(0.5),
                         sup, stot, round(sup/stot,4), p["cluster"], p["ns"], "sysdig", ts(t)])
            if len(rows) >= 2000: break
        if len(rows) >= 2000: break
    emit("ml_features", hdr, rows)

def g_pod_baselines():
    hdr = ["id","pod_name","namespace","cluster_name","metric_name","mean_value","stddev_value",
           "p95_value","min_value","max_value","sample_count","window_days","computed_at"]
    rows = []; i = 0
    for p in PODS:
        for mn in METRIC_NAMES:   # all metrics per pod → ≥1100 rows
            mean = between(10, 5000); i += 1
            rows.append([i, p["pod"], p["ns"], p["cluster"], mn, round(mean,3), round(mean*0.12,3),
                         round(mean*1.4,3), round(mean*0.5,3), round(mean*1.9,3),
                         rng.randint(500,5000), 7, ts(tbetween(NOW-timedelta(days=2), NOW))])
    emit("pod_baselines", hdr, rows)

# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 2 — ML / prediction (predictions feed alerts + feedback)
# ═══════════════════════════════════════════════════════════════════════════
PREDICTIONS = []  # shared: {id, uuid, pod, use_case, prob, risk, ts, env}
RISK = lambda pr: "critical" if pr>0.85 else "high" if pr>0.65 else "medium" if pr>0.4 else "low"
UC_BY_SCEN = {"memory_leak":[1,2,10],"oom_crashloop":[2,4,1],"cpu_throttle":[3,13],
              "disk_pressure":[5],"network_latency":[7,8,13],"node_pressure":[6,14],"healthy":[1,3,9,13]}

def g_predictions():
    hdr = ["id","prediction_id","pod_name","namespace","cluster_name","environment","use_case_id",
           "use_case_name","leak_probability","risk_level","confidence_score","lstm_prediction",
           "online_rf_prediction","isolation_forest_prediction","anomaly_detected","anomaly_score",
           "time_to_impact_seconds","projected_peak","optimal_action_window","worker_agent_id",
           "cluster_manager_id","strategist_agent_id","escalation_needed","escalated_at","action_required",
           "action_taken","action_result","prediction_timestamp","created_at"]
    rows = []; i = 0
    for _ in range(1500):
        incident = chance(0.7) and INCIDENT_PODS
        p = pick(INCIDENT_PODS) if incident else pick(PODS)
        ucid = pick(UC_BY_SCEN.get(p["scenario"], [1,3,13]))
        uc = next(u for u in USE_CASES if u[0]==ucid)
        base = between(0.55, 0.97) if p["scenario"]!="healthy" else between(0.02, 0.45)
        pr = round(base, 4); risk = RISK(pr)
        t = tbetween(p["onset"] or START, NOW)
        wagent = pick([a for a in AGENTS if a["cluster"]==p["cluster"] and a["type"]=="worker"])
        mgr = pick([a for a in AGENTS if a["cluster"]==p["cluster"] and a["type"]=="cluster_manager"])
        strat = pick([a for a in AGENTS if a["type"]=="strategist"])
        esc = pr>0.8; act = pr>0.65
        pid = U(); i += 1
        PREDICTIONS.append({"id":i,"uuid":pid,"pod":p,"ucid":ucid,"uc":uc[1],"prob":pr,"risk":risk,"ts":t,"env":p["env"]})
        rows.append([i, pid, p["pod"], p["ns"], p["cluster"], p["env"], ucid, uc[1], pr, risk,
            round(between(0.6,0.99),4), round(min(0.99,pr+between(-0.1,0.1)),4),
            round(min(0.99,pr+between(-0.12,0.12)),4), round(between(0.0,0.9),4),
            pr>0.6, round(between(0.0,0.4),6), int(between(120, 7200)) if act else None,
            round(between(60,99),2), pick(["5m","15m","30m","1h","immediate"]),
            wagent["agent_id"], mgr["agent_id"], strat["agent_id"], esc,
            ts(t+timedelta(minutes=2)) if esc else None, act,
            pick(["restart_pod","scale_up","increase_limit","drain_node","none"]) if act else None,
            pick(["success","success","pending","failed"]) if act else None, ts(t), ts(t)])
    emit("predictions", hdr, rows)

def g_alert_log():
    hdr = ["id","prediction_id","use_case_id","namespace","pod_name","cluster_name","gbm_prob",
           "risk_level","delivered","channel","suppressed","suppress_reason","alerted_at"]
    rows = []; i = 0
    for pr in [x for x in PREDICTIONS if x["prob"]>0.6][:1300]:
        supp = chance(0.12); i += 1
        rows.append([i, pr["id"], pr["ucid"], pr["pod"]["ns"], pr["pod"]["pod"], pr["pod"]["cluster"],
                     pr["prob"], pr["risk"], not supp, pick(["teams","teams","slack","pagerduty","email"]),
                     supp, "duplicate within suppression window" if supp else None,
                     ts(pr["ts"]+timedelta(seconds=rng.randint(5,90)))])
    # pad with extra lower-risk alerts to clear 1100 if needed
    while len(rows) < 1100:
        pr = pick(PREDICTIONS); i += 1
        rows.append([i, pr["id"], pr["ucid"], pr["pod"]["ns"], pr["pod"]["pod"], pr["pod"]["cluster"],
                     pr["prob"], pr["risk"], True, "teams", False, None,
                     ts(pr["ts"]+timedelta(seconds=rng.randint(5,90)))])
    emit("alert_log", hdr, rows)

EXPERTS = [("achen","sre-oncall"),("mpatel","sre-lead"),("jkim","platform-eng"),
           ("lwong","sre-oncall"),("rsingh","sre-lead"),("tgarcia","platform-eng")]
def g_expert_feedback():
    hdr = ["id","feedback_id","prediction_id","pod_name","environment","expert_username","expert_role",
           "accuracy_rating","usefulness_rating","timing_rating","remediation_quality","actual_leak_occurred",
           "actual_time_to_leak_seconds","action_taken","action_effective","prevented_outage","expert_comments",
           "suggested_improvements","false_positive_reason","false_negative_reason","model_should_learn",
           "confidence_adjustment","feedback_timestamp","created_at","feedback_notes"]
    rows = []; i = 0
    for pr in rng.sample(PREDICTIONS, k=min(len(PREDICTIONS), 1150)):
        u, role = pick(EXPERTS)
        leaked = pr["prob"]>0.6 and chance(0.8)
        fp = pr["prob"]>0.6 and not leaked
        acc = rng.randint(4,5) if (leaked or pr["prob"]<0.4) else rng.randint(1,3)
        i += 1
        rows.append([i, U(), pr["uuid"], pr["pod"]["pod"], pr["env"], u, role, acc, rng.randint(2,5),
            rng.randint(2,5), rng.randint(2,5), leaked, int(between(120,7200)) if leaked else None,
            pick(["restarted pod","scaled deployment","raised memory limit","drained node","monitored"]),
            chance(0.8), leaked and chance(0.7),
            pick(["Caught the leak ~20m early, gave us time to scale.","Accurate call on payments-api.",
                  "Alert fired but pod self-recovered.","Spot-on, prevented an outage.",
                  "Useful but timing was a bit late."]),
            pick([None,"tune threshold for staging","add node-pressure feature","longer lookback window"]),
            "metric spike was a deploy, not a leak" if fp else None,
            None, not fp or chance(0.5), round(between(-0.2,0.2),2),
            ts(pr["ts"]+timedelta(hours=rng.randint(1,48))), ts(NOW),
            pick([None,"follow-up in retro","linked to INC-"+str(rng.randint(4000,4999))])])
    emit("expert_feedback", hdr, rows)

def g_ml_use_cases():
    hdr = ["id","name","priority","category","description","created_at"]
    rows = [[u[0],u[1],u[2],u[3],u[4],ts(START)] for u in USE_CASES]
    emit("ml_use_cases", hdr, rows)

def g_model_performance():
    hdr = ["id","model_name","model_version","environment","accuracy","precision_score","recall","f1_score",
           "auc_score","false_positive_rate","false_negative_rate","use_case_id","use_case_accuracy",
           "evaluation_start","evaluation_end","sample_size","training_data_size","created_at","promoted",
           "model_type","dkubex_job_id","model_path","trained_at","training_seconds","training_rows"]
    rows = []; i = 0
    models = ["gbm","lstm","online_rf","isolation_forest"]
    for _ in range(1100):
        mt = pick(models); ucid = pick([u[0] for u in USE_CASES]); env = pick(ENVIRONMENTS)
        auc = round(between(0.78, 0.97),4); t0 = tbetween(START, NOW)
        i += 1
        rows.append([i, mt, f"v{rng.randint(1,9)}.{rng.randint(0,9)}", env, round(between(0.8,0.97),4),
            round(between(0.75,0.96),4), round(between(0.7,0.95),4), round(between(0.74,0.95),4), auc,
            round(between(0.01,0.12),4), round(between(0.02,0.15),4), ucid, round(between(0.7,0.96),4),
            ts(t0), ts(t0+timedelta(hours=2)), rng.randint(5000,80000), rng.randint(20000,300000),
            ts(t0), chance(0.3), mt, f"dkubex-job-{rng.randint(10000,99999)}",
            f"/models/{mt}/{env}/v{rng.randint(1,9)}", ts(t0), round(between(120,5400),1), rng.randint(20000,300000)])
    emit("model_performance", hdr, rows)

def g_model_promotion_log():
    hdr = ["id","new_auc","current_auc","promoted","notes","evaluated_at"]
    rows = []; i = 0
    for _ in range(1100):
        cur = between(0.78,0.93); new = cur + between(-0.05,0.06); prom = new>cur
        i += 1
        rows.append([i, round(new,4), round(cur,4), prom,
            ("promoted: +%.3f AUC" % (new-cur)) if prom else "held: no AUC gain",
            ts(tbetween(START, NOW))])
    emit("model_promotion_log", hdr, rows)

def g_prediction_schedule():
    hdr = ["id","cycle_num","started_at","finished_at","passed","predictions_written","use_cases_run",
           "error_message","created_at"]
    rows = []; i = 0
    t = START
    for c in range(1, 1200):
        i += 1; dur = rng.randint(20, 180); ok = chance(0.95)
        rows.append([i, c, ts(t), ts(t+timedelta(seconds=dur)), ok, rng.randint(20,400) if ok else 0,
                     [str(x) for x in rng.sample([u[0] for u in USE_CASES], k=rng.randint(3,8))],
                     None if ok else pick(["sysdig timeout","db connection reset","feature build failed"]), ts(t)])
        t += timedelta(minutes=rng.randint(15,40))
    emit("prediction_schedule", hdr, rows)

# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 3 — Multi-agent + DAG orchestration
# ═══════════════════════════════════════════════════════════════════════════
def g_agents():
    hdr = ["id","agent_id","agent_type","agent_layer","cluster_group","environment","status",
           "last_heartbeat","uptime_seconds","messages_processed","predictions_made","escalations_sent",
           "actions_executed","created_at","updated_at"]
    rows = []; i = 0
    # base fleet + historical instances to exceed 1100 unique agent_ids
    base = list(AGENTS)
    extra = [{"agent_id":f"{a['type']}-{a['cluster']}-h{n:03d}","type":a["type"],"layer":a["layer"],
              "cluster":a["cluster"],"env":a["env"]} for a in base for n in range((1200//len(base))+1)]
    for a in (base+extra)[:1200]:
        i += 1; up = rng.randint(3600, 2_600_000)
        rows.append([i, a["agent_id"], a["type"], a["layer"], a["cluster"], a["env"],
                     pick(["active","active","active","degraded","stopped"]),
                     ts(NOW-timedelta(seconds=rng.randint(0,300))), up, rng.randint(1000,500000),
                     rng.randint(0,40000), rng.randint(0,4000), rng.randint(0,8000),
                     ts(NOW-timedelta(seconds=up)), ts(NOW)])
    emit("agents", hdr, rows)

def g_agent_status():
    hdr = ["id","agent_name","agent_type","status","cluster_name","last_heartbeat","metadata","created_at","updated_at"]
    rows = []; i = 0
    for _ in range(1100):
        a = pick(AGENTS); i += 1
        rows.append([i, a["agent_id"], a["type"], pick(["active","active","degraded","idle"]),
                     a["cluster"], ts(NOW-timedelta(seconds=rng.randint(0,600))),
                     {"version":"1.4.2","queue_depth":rng.randint(0,50),"load":round(between(0,1),2)},
                     ts(tbetween(START,NOW)), ts(NOW)])
    emit("agent_status", hdr, rows)

MSG_TYPES = ["prediction","escalation","action_request","action_result","heartbeat","query","ack"]
def g_agent_communications():
    hdr = ["id","correlation_id","source_agent_id","target_agent_id","message_type","message_data",
           "priority","processing_time_ms","success","error_message","retry_count","sent_at","processed_at","environment"]
    rows = []; i = 0
    for _ in range(1600):
        s = pick(AGENTS); tg = pick([a for a in AGENTS if a["agent_id"]!=s["agent_id"]])
        ok = chance(0.95); t = tbetween(START, NOW); mt = pick(MSG_TYPES); i += 1
        rows.append([i, U(), s["agent_id"], tg["agent_id"], mt,
            {"type":mt,"pod":pick(PODS)["pod"],"prob":round(between(0,1),3)},
            pick(["low","normal","normal","high","critical"]), rng.randint(2,800), ok,
            None if ok else pick(["timeout","target unavailable","schema mismatch"]),
            0 if ok else rng.randint(1,3), ts(t), ts(t+timedelta(milliseconds=rng.randint(2,800))), s["env"]])
    emit("agent_communications", hdr, rows)

def g_agent_messages():
    hdr = ["id","correlation_id","source","target","message_type","payload","status","created_at","processed_at"]
    rows = []; i = 0
    for _ in range(1300):
        s = pick(AGENTS); tg = pick(AGENTS); t = tbetween(START, NOW)
        st = pick(["processed","processed","pending","failed"]); i += 1
        rows.append([i, U(), s["agent_id"], tg["agent_id"], pick(MSG_TYPES),
                     {"pod":pick(PODS)["pod"],"action":pick(["restart","scale","notify"])}, st, ts(t),
                     ts(t+timedelta(seconds=rng.randint(0,30))) if st!="pending" else None])
    emit("agent_messages", hdr, rows)

def g_action_progress():
    hdr = ["id","action_id","action_type","cluster_name","agent_name","status","progress_percentage",
           "steps","metadata","started_at","completed_at","created_at","updated_at"]
    rows = []; i = 0
    actions = ["restart_pod","scale_deployment","increase_memory_limit","drain_node","rollback_deploy","cordon_node"]
    for _ in range(1100):
        p = pick(PODS); at = pick(actions); st = pick(["completed","completed","running","failed","pending"])
        prog = 100 if st=="completed" else 0 if st=="pending" else rng.randint(10,90)
        t = tbetween(START, NOW); i += 1
        steps = [{"step":s,"done":prog>=(k+1)*25} for k,s in enumerate(["plan","validate","execute","verify"])]
        rows.append([i, f"act-{U()[:8]}", at, p["cluster"], pick(AGENTS)["agent_id"], st, prog,
                     steps, {"pod":p["pod"],"namespace":p["ns"],"reason":p["scenario"]}, ts(t),
                     ts(t+timedelta(minutes=rng.randint(1,20))) if st=="completed" else None, ts(t), ts(NOW)])
    emit("action_progress", hdr, rows)

DAG_IDS = []
def g_dag_instances():
    hdr = ["dag_id","correlation_id","dag_type","trigger_event","status","created_at","started_at",
           "completed_at","duration_ms","environment","cluster_name","namespace","pod_name","metadata",
           "tags","node_count","edge_count","success_rate","updated_at"]
    rows = []
    dtypes = ["remediation","prediction_pipeline","escalation","capacity_planning","incident_response"]
    for _ in range(1150):
        p = pick(PODS); dt = pick(dtypes); did = f"dag-{U()[:12]}"
        t = tbetween(START, NOW); dur = rng.randint(500, 60000); st = pick(["COMPLETED","COMPLETED","RUNNING","FAILED"])
        nc = rng.randint(3, 9); ec = nc - 1 + rng.randint(0,3)
        DAG_IDS.append({"dag_id":did,"env":p["env"],"dtype":dt,"t":t,"status":st,"nc":nc})
        rows.append([did, U(), dt, {"event":p["scenario"],"pod":p["pod"],"prob":round(between(0,1),3)}, st,
                     ts(t), ts(t+timedelta(seconds=2)), ts(t+timedelta(milliseconds=dur)) if st!="RUNNING" else None,
                     dur if st!="RUNNING" else None, p["env"], p["cluster"], p["ns"], p["pod"],
                     {"priority":pick(["p1","p2","p3"])}, [dt, p["ns"], pick(["auto","manual"])], nc, ec,
                     round(between(0.6,1.0),4), ts(NOW)])
    emit("dag_instances", hdr, rows)

NODE_IDS = []
def g_dag_nodes():
    hdr = ["node_id","dag_id","node_type","task_name","agent_name","agent_layer","status","created_at",
           "started_at","completed_at","duration_ms","input_data","output_data","error_message",
           "retry_count","max_retries","cpu_usage_ms","memory_usage_mb","metadata","updated_at"]
    rows = []
    tasks = ["collect_metrics","run_prediction","evaluate_risk","match_sop","decide_action","execute_action","verify_outcome","notify"]
    for d in DAG_IDS:
        for k in range(d["nc"]):
            nid = f"node-{U()[:12]}"; a = pick(AGENTS); st = "COMPLETED" if d["status"]=="COMPLETED" else pick(["COMPLETED","RUNNING","FAILED","PENDING"])
            t = d["t"]+timedelta(seconds=k*3); dur = rng.randint(50, 8000)
            NODE_IDS.append({"node_id":nid,"dag_id":d["dag_id"]})
            rows.append([nid, d["dag_id"], pick(["task","decision","action","io"]), pick(tasks),
                         a["agent_id"], a["layer"], st, ts(t), ts(t+timedelta(milliseconds=20)),
                         ts(t+timedelta(milliseconds=dur)) if st in ("COMPLETED","FAILED") else None,
                         dur if st in ("COMPLETED","FAILED") else None, {"k":k}, {"ok":st=="COMPLETED"},
                         None if st!="FAILED" else pick(["timeout","exception","invalid input"]),
                         0 if st!="FAILED" else rng.randint(1,3), 3, rng.randint(5,4000), rng.randint(16,512),
                         {"layer":a["layer"]}, ts(NOW)])
    emit("dag_nodes", hdr, rows)

def g_dag_edges():
    hdr = ["edge_id","dag_id","from_node_id","to_node_id","edge_type","condition_expression","condition_met",
           "created_at","activated_at","data_transferred","transfer_size_bytes","metadata"]
    rows = []
    by_dag = {}
    for n in NODE_IDS: by_dag.setdefault(n["dag_id"], []).append(n["node_id"])
    for did, nlist in by_dag.items():
        for j in range(len(nlist)-1):
            t = tbetween(START, NOW)
            rows.append([f"edge-{U()[:12]}", did, nlist[j], nlist[j+1],
                         pick(["DATA_FLOW","DATA_FLOW","CONTROL","CONDITIONAL"]),
                         pick([None,"risk>0.65","action_required==true","sop_matched"]), chance(0.9),
                         ts(t), ts(t+timedelta(milliseconds=10)), {"bytes":rng.randint(50,5000)},
                         rng.randint(50,5000), {"hop":j}])
    emit("dag_edges", hdr, rows)

def g_dag_execution_history():
    hdr = ["id","dag_id","dag_type","environment","total_duration_ms","node_count","success_rate","final_status",
           "actions_executed","predictions_made","errors_encountered","avg_node_duration_ms","max_node_duration_ms",
           "total_cpu_usage_ms","total_memory_usage_mb","trigger_summary","outcome_summary","execution_date","created_at"]
    rows = []; i = 0
    for d in DAG_IDS:
        i += 1; dur = rng.randint(500, 60000)
        rows.append([i, d["dag_id"], d["dtype"], d["env"], dur, d["nc"], round(between(0.6,1.0),4),
                     d["status"], rng.randint(0,5), rng.randint(0,8), rng.randint(0,3),
                     int(dur/max(1,d["nc"])), int(dur*0.4), rng.randint(100,20000), rng.randint(64,2048),
                     {"trigger":d["dtype"]}, {"resolved":d["status"]=="COMPLETED"},
                     d["t"].strftime("%Y-%m-%d"), ts(d["t"])])
    emit("dag_execution_history", hdr, rows)

# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 4 — SOP / knowledge (RAG)
# ═══════════════════════════════════════════════════════════════════════════
def g_sop_registry():
    hdr = ["sop_path","sop_name","domain","status","file_hash","version","chunk_count","first_ingested_at",
           "last_ingested_at","last_hash_change_at","deprecated_at","deprecation_reason","replaced_by"]
    rows = []
    for s in SOPS:
        dep = chance(0.1); t0 = tbetween(START-timedelta(days=60), START)
        rows.append([s["path"], s["name"], s["domain"], "deprecated" if dep else "active",
                     hashlib.md5(s["path"].encode()).hexdigest(), rng.randint(1,5), rng.randint(3,25),
                     ts(t0), ts(tbetween(t0, NOW)), ts(tbetween(t0, NOW)),
                     ts(NOW-timedelta(days=rng.randint(1,20))) if dep else None,
                     "superseded by newer runbook" if dep else None,
                     pick([s2["path"] for s2 in SOPS]) if dep else None])
    emit("sop_registry", hdr, rows)

def g_sop_effectiveness():
    hdr = ["sop_id","sop_name","sop_path","domain","total_matches","avg_similarity","max_similarity",
           "times_actioned","useful_count","not_useful_count","success_rate","first_matched_at",
           "last_matched_at","staleness_days","status","updated_at"]
    rows = []
    for s in SOPS:
        tm = rng.randint(5, 800); useful = rng.randint(0, tm); nu = tm-useful
        rows.append([s["id"], s["name"], s["path"], s["domain"], tm, round(between(0.6,0.9),4),
                     round(between(0.9,0.99),4), rng.randint(0,tm), useful, nu,
                     round(useful/max(1,tm),4), ts(tbetween(START-timedelta(days=40),START)),
                     ts(tbetween(START,NOW)), rng.randint(0,40), pick(["active","active","stale"]), ts(NOW)])
    emit("sop_effectiveness", hdr, rows)

GAP_CLUSTER_IDS = []
def g_sop_gap_clusters():
    hdr = ["id","cluster_label","gap_category","gap_count","representative_query","centroid_embedding",
           "avg_similarity","first_seen_at","last_seen_at","status","draft_sop_id","reviewed","created_at","updated_at"]
    rows = []; i = 0
    cats = ["memory","cpu","network","storage","node","security","availability"]
    for _ in range(1100):
        i += 1; cat = pick(cats); st = pick(["open","open","draft_created","resolved","ignored"])
        GAP_CLUSTER_IDS.append(i)
        rows.append([i, f"{cat} gaps cluster {i}", cat, rng.randint(2,40),
                     f"how to remediate {cat} issue on {pick(PODS)['svc']}",
                     [round(between(-1,1),3) for _ in range(8)], round(between(0.3,0.6),4),
                     ts(tbetween(START,NOW)), ts(tbetween(START,NOW)), st, None, chance(0.4), ts(START), ts(NOW)])
    emit("sop_gap_clusters", hdr, rows)

GAP_IDS = []
def g_sop_gaps():
    hdr = ["id","query_text","best_match_sop","best_similarity","source","cluster_name","namespace",
           "pod_name","gap_category","gap_cluster_id","reviewed","draft_sop_created","detected_at"]
    rows = []; i = 0
    for _ in range(1400):
        p = pick(PODS); s = pick(SOPS); i += 1; GAP_IDS.append(i)
        rows.append([i, f"why is {p['svc']} {pick(['restarting','throttled','slow','OOMKilled'])} in {p['ns']}?",
                     s["path"], round(between(0.2,0.55),4), pick(["chat","alert","prediction","auto"]),
                     p["cluster"], p["ns"], p["pod"], pick(["memory","cpu","network","storage","node"]),
                     pick(GAP_CLUSTER_IDS), chance(0.4), chance(0.2), ts(tbetween(START,NOW))])
    emit("sop_gaps", hdr, rows)

def g_sop_gap_members():
    hdr = ["id","cluster_id","gap_id","distance","assigned_at"]
    rows = []; i = 0
    for gid in GAP_IDS[:1100]:
        i += 1
        rows.append([i, pick(GAP_CLUSTER_IDS), gid, round(between(0.0,0.8),4), ts(tbetween(START,NOW))])
    emit("sop_gap_members", hdr, rows)

def g_sop_drafts():
    hdr = ["id","cluster_id","suggested_filename","suggested_domain","title","content","generation_method",
           "gap_count","representative_queries","status","reviewed_by","reviewed_at","rejection_reason",
           "ingested_at","ingested_sop_path","created_at","updated_at"]
    rows = []; i = 0
    for _ in range(1100):
        i += 1; cat = pick(["memory","cpu","network","storage","node","security"]); st = pick(["draft","draft","approved","rejected","ingested"])
        rows.append([i, pick(GAP_CLUSTER_IDS), f"runbook-{cat}-{i}.md", cat,
            f"Auto-draft: remediating {cat} incidents",
            f"# {cat.title()} remediation\n\n## Symptoms\n- elevated {cat} signals\n\n## Steps\n1. Inspect pod\n2. Apply fix\n3. Verify",
            pick(["auto","auto","llm"]), rng.randint(2,30), [f"{cat} query {k}" for k in range(rng.randint(1,4))],
            st, pick([None,"mpatel","rsingh"]), ts(tbetween(START,NOW)) if st in ("approved","rejected","ingested") else None,
            "too generic" if st=="rejected" else None,
            ts(tbetween(START,NOW)) if st=="ingested" else None,
            f"sops/{cat}/runbook-{cat}-{i}.md" if st=="ingested" else None, ts(START), ts(NOW)])
    emit("sop_drafts", hdr, rows)

def g_sop_usage_tracking():
    hdr = ["id","sop_id","sop_name","sop_path","section_title","query_text","similarity_score","source",
           "cluster_name","namespace","pod_name","action_taken","was_useful","matched_at"]
    rows = []; i = 0
    for _ in range(1500):
        s = pick(SOPS); p = pick(PODS); i += 1
        rows.append([i, s["id"], s["name"], s["path"], pick(["Symptoms","Remediation","Prevention","Diagnosis"]),
                     f"{p['svc']} {pick(['OOMKilled','CrashLoopBackOff','high latency','throttled'])}",
                     round(between(0.6,0.95),4), pick(["chat","alert","auto","prediction"]), p["cluster"],
                     p["ns"], p["pod"], pick(["restart_pod","scale_up","increase_limit","none"]),
                     chance(0.7), ts(tbetween(START,NOW))])
    emit("sop_usage_tracking", hdr, rows)

def g_knowledge_base():
    hdr = ["id","knowledge_type","title","content","source","confidence","applicable_environments","tags",
           "metadata","created_at","updated_at"]
    rows = []; i = 0
    ktypes = ["incident_pattern","remediation","root_cause","best_practice","postmortem"]
    for _ in range(1100):
        kt = pick(ktypes); dom = pick(["memory","cpu","network","storage","node","availability"]); i += 1
        rows.append([i, kt, f"{dom.title()} {kt.replace('_',' ')} #{i}",
            f"When {dom} pressure rises on a pod, the typical root cause is undersized limits or a leak; "
            f"remediate by right-sizing requests/limits, adding a PDB, and tuning probes.",
            pick(["expert_feedback","postmortem","auto_mined","sop"]), round(between(0.6,0.98),4),
            rng.sample(ENVIRONMENTS, k=rng.randint(1,3)), rng.sample([dom,"k8s","sre","prod","leak","scaling"], k=3),
            {"domain":dom,"verified":chance(0.7)}, ts(tbetween(START-timedelta(days=60),NOW)), ts(NOW)])
    emit("knowledge_base", hdr, rows)

def g_pattern_signatures():
    hdr = ["id","pattern_type","pattern_name","cluster_name","namespace","signature_vector","confidence",
           "occurrence_count","first_seen","last_seen","metadata","created_at"]
    rows = []; i = 0
    for _ in range(1100):
        p = pick(PODS); i += 1
        rows.append([i, pick(["memory_leak","cpu_spike","crashloop","latency","oom"]),
                     f"{p['svc']}-{pick(['sawtooth','rampup','spike','plateau'])}", p["cluster"], p["ns"],
                     [round(between(0,1),3) for _ in range(12)], round(between(0.5,0.97),4),
                     rng.randint(1,200), ts(tbetween(START-timedelta(days=30),START)), ts(tbetween(START,NOW)),
                     {"svc":p["svc"],"scenario":p["scenario"]}, ts(NOW)])
    emit("pattern_signatures", hdr, rows)

# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 5 — pipeline / collection / ops
# ═══════════════════════════════════════════════════════════════════════════
def g_sysdig_queries():
    hdr = ["id","name","query_text","priority","interval_seconds","ml_use_cases","description","created_at"]
    rows = []; i = 0
    qs = [("mem_working_set","container_memory_working_set_bytes","P1",30,[1,2,10]),
          ("cpu_usage","rate(container_cpu_usage_seconds_total[5m])","P2",30,[3]),
          ("restart_total","kube_pod_container_status_restarts_total","P1",60,[4]),
          ("http_5xx","rate(http_requests_total{code=~\"5..\"}[5m])","P2",30,[13]),
          ("net_latency","histogram_quantile(0.99, http_request_duration_ms)","P2",30,[7]),
          ("fs_usage","container_fs_usage_bytes","P2",60,[5]),
          ("node_ready","kube_node_status_condition{condition=\"Ready\"}","P1",60,[6]),
          ("gc_pause","go_gc_duration_seconds","P3",60,[10]),
          ("dns_latency","coredns_dns_request_duration_seconds","P2",30,[8]),
          ("tcp_conns","container_network_tcp_usage_total","P3",60,[12])]
    for n,(nm,q,pr,iv,uc) in enumerate(qs*3):  # variants
        i += 1; suffix = "" if n < len(qs) else f"_v{n//len(qs)+1}"
        rows.append([i, nm+suffix, q, pr, iv, uc, f"Sysdig query for {nm}", ts(START)])
    emit("sysdig_queries", hdr, rows)

def g_sysdig_query_results():
    hdr = ["id","query_name","status","result_count","sample_data","tested_at"]
    rows = []; i = 0
    for _ in range(1100):
        i += 1; ok = chance(0.9)
        rows.append([i, pick(["mem_working_set","cpu_usage","http_5xx","restart_total","net_latency"]),
                     "success" if ok else pick(["timeout","error","no_data"]), rng.randint(0,500) if ok else 0,
                     {"sample":[round(between(0,1e5),2) for _ in range(3)]}, ts(tbetween(START,NOW))])
    emit("sysdig_query_results", hdr, rows)

def g_source_registry():
    hdr = ["id","source_name","source_type","enabled","priority","last_collection_at","last_row_count",
           "total_rows_collected","total_rows_rejected","total_rows_deduped","status","error_message",
           "created_at","updated_at"]
    srcs = [("sysdig","metrics",True,1),("prometheus","metrics",True,2),("kube_events","events",True,3),
            ("kube_state","state",True,4),("loki","logs",False,5),("cloudwatch","metrics",False,6),
            ("otel","traces",True,7),("custom_exporter","metrics",False,8)]
    rows = []
    for n,(nm,tp,en,pr) in enumerate(srcs, 1):
        rows.append([n, nm, tp, en, pr, ts(NOW-timedelta(seconds=rng.randint(0,300))), rng.randint(0,5000),
                     rng.randint(100000,9000000), rng.randint(0,50000), rng.randint(0,80000),
                     "active" if en else "inactive", None if en else "disabled by operator",
                     ts(START-timedelta(days=90)), ts(NOW)])
    emit("source_registry", hdr, rows)

def g_collection_summaries():
    hdr = ["id","timestamp","successful_queries","total_queries","total_metrics_collected","success_rate",
           "collection_duration_seconds","summary_data","created_at"]
    rows = []; i = 0; t = START
    for _ in range(1200):
        i += 1; tot = rng.randint(20,40); succ = tot-rng.randint(0,4)
        rows.append([i, ts(t), succ, tot, rng.randint(2000,50000), round(succ/tot,4),
                     round(between(5,90),2), {"by_source":{"sysdig":rng.randint(1000,30000)}}, ts(t)])
        t += timedelta(minutes=rng.randint(20,40))
    emit("collection_summaries", hdr, rows)

def g_dedup_log():
    hdr = ["id","timestamp","source_kept","source_discarded","metric_name","pod_name","namespace",
           "cluster_name","value_kept","value_discarded","count","created_at"]
    rows = []; i = 0
    for _ in range(1200):
        p = pick(PODS); v = between(0,1e5); i += 1
        rows.append([i, ts(tbetween(START,NOW)), pick(["sysdig","prometheus"]), pick(["prometheus","cloudwatch"]),
                     pick(METRIC_NAMES), p["pod"], p["ns"], p["cluster"], round(v,3), round(v*between(0.9,1.1),3),
                     rng.randint(1,20), ts(NOW)])
    emit("dedup_log", hdr, rows)

def g_sanitisation_log():
    hdr = ["id","timestamp","source","reason","metric_name","raw_value","pod_name","namespace","count","created_at"]
    rows = []; i = 0
    for _ in range(1200):
        p = pick(PODS); i += 1
        rows.append([i, ts(tbetween(START,NOW)), pick(["sysdig","prometheus","cloudwatch"]),
                     pick(["null_value","negative_value","out_of_range","nan","stale_timestamp"]),
                     pick(METRIC_NAMES), pick(["NaN","-1","null","1e309",""]), p["pod"], p["ns"],
                     rng.randint(1,30), ts(NOW)])
    emit("sanitisation_log", hdr, rows)

def g_metrics_cache():
    hdr = ["cache_key","cache_value","created_at","expires_at"]
    rows = []
    for _ in range(1100):
        p = pick(PODS); t = tbetween(NOW-timedelta(hours=6),NOW)
        rows.append([f"mc:{p['cluster']}:{p['ns']}:{p['pod']}:{pick(METRIC_NAMES)}",
                     {"value":round(between(0,1e5),2),"ts":ts(t)}, ts(t), ts(t+timedelta(minutes=5))])
    emit("metrics_cache", hdr, rows)

def g_prediction_cache():
    hdr = ["cache_key","cache_value","created_at","expires_at"]
    rows = []
    for pr in (PREDICTIONS[:1100] or []):
        rows.append([f"pc:{pr['pod']['pod']}:{pr['ucid']}", {"prob":pr["prob"],"risk":pr["risk"]},
                     ts(pr["ts"]), ts(pr["ts"]+timedelta(minutes=5))])
    while len(rows) < 1100:
        p = pick(PODS)
        rows.append([f"pc:{p['pod']}:{rng.randint(1,15)}", {"prob":round(between(0,1),3)}, ts(NOW), ts(NOW+timedelta(minutes=5))])
    emit("prediction_cache", hdr, rows)

def g_real_data_test():
    hdr = ["id","timestamp","test_data","created_at"]
    rows = []; i = 0
    for _ in range(1100):
        p = pick(PODS); i += 1
        rows.append([i, ts(tbetween(START,NOW)), {"pod":p["pod"],"ns":p["ns"],"sample":round(between(0,1),4)}, ts(NOW)])
    emit("real_data_test", hdr, rows)

# ═══════════════════════════════════════════════════════════════════════════
#  RUN — order respects FK dependencies
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\ngenerating CSVs -> out/")
    # group 1
    g_kubernetes_pod_status(); g_kubernetes_node_status(); g_kubernetes_events()
    g_comprehensive_metrics(); g_metrics_realtime(); g_metrics_compressed(); g_sysdig_metrics()
    g_ml_features(); g_pod_baselines()
    # group 2 (predictions first; alerts/feedback depend on it)
    g_predictions(); g_alert_log(); g_expert_feedback(); g_ml_use_cases()
    g_model_performance(); g_model_promotion_log(); g_prediction_schedule()
    # group 3 (dag_instances -> nodes -> edges)
    g_agents(); g_agent_status(); g_agent_communications(); g_agent_messages(); g_action_progress()
    g_dag_instances(); g_dag_nodes(); g_dag_edges(); g_dag_execution_history()
    # group 4 (gap_clusters/gaps before members/drafts)
    g_sop_registry(); g_sop_effectiveness(); g_sop_gap_clusters(); g_sop_gaps()
    g_sop_gap_members(); g_sop_drafts(); g_sop_usage_tracking(); g_knowledge_base(); g_pattern_signatures()
    # group 5
    g_sysdig_queries(); g_sysdig_query_results(); g_source_registry(); g_collection_summaries()
    g_dedup_log(); g_sanitisation_log(); g_metrics_cache(); g_prediction_cache(); g_real_data_test()

    tables = len(SUMMARY); total = sum(SUMMARY.values())
    print(f"\nDONE: {tables} tables, {total:,} rows -> {OUT}")
    under = {k:v for k,v in SUMMARY.items() if v < 1100}
    if under: print("tables under 1100 (natural dimensions):", under)

if __name__ == "__main__":
    main()

