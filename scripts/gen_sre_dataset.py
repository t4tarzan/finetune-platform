#!/usr/bin/env python3
"""
Generate the SRE / Kubernetes reliability fine-tuning datasets for the demo.

Schema (one JSON object per line — consumed by pipeline/train_qlora.prepare_training_data,
which maps {question, context, reference_answer} → {prompt, completion}):

    {
      "id":          "sre-00042",
      "category":    "CrashLoopBackOff",      # failure mode / signal class
      "severity":    "SEV2",
      "question":    "<SRE asks about a pod/cluster/uptime symptom>",
      "context":     "<kubectl-style status + events + log/uptime snippet>",
      "reference_answer": "Root cause: ...\nRemediation: ...\nPreventive measure: ...",
      "metadata":    {"service": "...", "namespace": "...", "signal": "..."}
    }

The dataset is deliberately about *predictable* pod-failure reasons and the
*preventive measures* that stop them recurring — the platform learns to turn raw
pod status / events / uptime logs into a root-cause + remediation + prevention answer.

Outputs:
    data/sre-pods/dataset_v1.jsonl   (100 rows  — first training run)
    data/sre-pods/dataset_v2.jsonl   (150 rows  — same 100 + 50 new rows that
                                       deepen the node-drain / CrashLoopBackOff
                                       "hero" scenario and its preventive measures)
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sre-pods")
os.makedirs(OUT_DIR, exist_ok=True)

SERVICES = [
    ("payments-api", "payments"), ("checkout-web", "storefront"),
    ("orders-worker", "orders"), ("inventory-svc", "warehouse"),
    ("auth-gateway", "identity"), ("search-api", "discovery"),
    ("notifications", "comms"), ("ledger-svc", "finance"),
    ("recommendation", "ml"), ("image-resizer", "media"),
    ("session-store", "platform"), ("billing-cron", "finance"),
]

rows = []
_seen = set()


def add(category, severity, question, context, answer, service, namespace, signal):
    rid = f"sre-{len(rows)+1:05d}"
    key = (category, question)
    if key in _seen:
        return
    _seen.add(key)
    rows.append({
        "id": rid,
        "category": category,
        "severity": severity,
        "question": question.strip(),
        "context": context.strip(),
        "reference_answer": answer.strip(),
        "metadata": {"service": service, "namespace": namespace, "signal": signal},
    })


# ── Failure-mode templates. Each returns (question, context, answer). ──────────

def crashloop(svc, ns):
    q = (f"Pods for {svc} in namespace {ns} keep restarting with status "
         f"CrashLoopBackOff. What is the root cause and how do we fix it?")
    ctx = (f"$ kubectl -n {ns} get pod\n"
           f"NAME                        READY   STATUS             RESTARTS   AGE\n"
           f"{svc}-7c9f5b8d4-2x      0/1     CrashLoopBackOff   7          11m\n"
           f"$ kubectl -n {ns} logs {svc}-7c9f5b8d4-2x --previous | tail -3\n"
           f"  panic: missing required env DATABASE_URL\n"
           f"  exit status 2\n"
           f"Events: Back-off restarting failed container")
    a = ("Root cause: the container exits non-zero on startup because a required "
         "configuration value (DATABASE_URL) is absent, so the kubelet restarts it "
         "with exponential back-off (CrashLoopBackOff).\n"
         "Remediation: add the missing key to the ConfigMap/Secret referenced by the "
         "Deployment, then `kubectl rollout restart deploy/" + svc + "`.\n"
         "Preventive measure: validate required env at boot and fail with a clear "
         "message; add a CI check that every referenced ConfigMap/Secret key exists, "
         "and a startupProbe so slow-but-healthy boots are not mistaken for crashes.")
    return q, ctx, a


def oomkilled(svc, ns):
    q = (f"{svc} pods in {ns} are being OOMKilled under load. Why, and what "
         f"preventive measure stops it recurring?")
    ctx = (f"$ kubectl -n {ns} describe pod {svc}-0 | grep -A2 'Last State'\n"
           f"    Last State:  Terminated\n"
           f"      Reason:    OOMKilled\n"
           f"      Exit Code: 137\n"
           f"resources.limits.memory: 256Mi    working_set peak: 412Mi")
    a = ("Root cause: the container's working set (peak 412Mi) exceeds its memory "
         "limit (256Mi); the kernel OOM-kills it, surfaced as exit code 137 / "
         "OOMKilled.\n"
         "Remediation: raise limits.memory to ~512Mi (with headroom above observed "
         "peak) and set requests.memory to the steady-state usage.\n"
         "Preventive measure: size requests/limits from real percentiles (p95+buffer), "
         "add a memory-usage alert at 80% of limit, and load-test before release so the "
         "ceiling is known rather than discovered in production.")
    return q, ctx, a


def imagepull(svc, ns):
    q = (f"A new rollout of {svc} in {ns} is stuck — pods show ImagePullBackOff. "
         f"What happened and how do we prevent it?")
    ctx = (f"$ kubectl -n {ns} get pod\n"
           f"{svc}-689c   0/1   ImagePullBackOff   0   3m\n"
           f"Events: Failed to pull image \"registry/{svc}:v2.3.1\": "
           f"manifest unknown")
    a = ("Root cause: the referenced image tag (registry/" + svc + ":v2.3.1) does not "
         "exist in the registry (manifest unknown), so the kubelet cannot start the "
         "container and backs off.\n"
         "Remediation: push/repair the tag or roll back to the last known-good image "
         "digest; verify imagePullSecrets if the registry is private.\n"
         "Preventive measure: deploy by immutable digest (not floating tags), gate "
         "rollouts on a registry existence check in CI, and keep the previous ReplicaSet "
         "for instant rollback.")
    return q, ctx, a


def readiness(svc, ns):
    q = (f"{svc} in {ns} shows Running but 0/1 READY and gets no traffic. What is "
         f"wrong and how should we prevent this class of incident?")
    ctx = (f"$ kubectl -n {ns} get pod\n"
           f"{svc}-5d7   1/1? 0/1   Running   0   6m\n"
           f"Events: Readiness probe failed: HTTP 503 on /healthz\n"
           f"$ kubectl -n {ns} get endpoints {svc}  ->  <none>")
    a = ("Root cause: the readiness probe (/healthz) returns 503, so the pod is kept "
         "out of the Service endpoints and receives no traffic even though the process "
         "is Running.\n"
         "Remediation: fix the dependency the health check reports unhealthy (often a DB "
         "or cache connection); confirm the probe path/port match the app.\n"
         "Preventive measure: make readiness reflect real dependency health, set a "
         "sensible initialDelaySeconds/periodSeconds, and alert when ready replicas < "
         "desired for more than a minute.")
    return q, ctx, a


def liveness(svc, ns):
    q = (f"{svc} pods in {ns} restart every few minutes even though the app looks "
         f"healthy. The liveness probe seems involved — what's the fix?")
    ctx = (f"Events: Liveness probe failed: timeout 1s exceeded; Killing container\n"
           f"liveness: httpGet /healthz timeoutSeconds=1 periodSeconds=5\n"
           f"p99 /healthz latency under GC: 1.4s")
    a = ("Root cause: the liveness probe timeout (1s) is shorter than the endpoint's "
         "p99 latency (1.4s) during GC pauses, so healthy pods fail the probe and are "
         "killed and restarted.\n"
         "Remediation: raise timeoutSeconds and failureThreshold so transient slowness "
         "doesn't trip it; keep the liveness handler cheap and dependency-free.\n"
         "Preventive measure: separate liveness (process alive) from readiness "
         "(can serve), and tune probe budgets from observed p99, not guesses.")
    return q, ctx, a


def node_notready(svc, ns):
    q = (f"Several {svc} pods in {ns} went to status NodeLost / Unknown at the same "
         f"time. What does the cluster state tell us and how do we prevent outages?")
    ctx = (f"$ kubectl get nodes\n"
           f"ip-10-2-3-4   NotReady   <none>   84d   v1.29\n"
           f"Conditions: KubeletNotReady (PLEG is not healthy)\n"
           f"$ uptime on node -> load average: 38.2, 30.1, 22.0")
    a = ("Root cause: the node is NotReady (kubelet/PLEG unhealthy under load average "
         "38), so its pods are marked Unknown and eventually evicted — a node-level "
         "failure, not an app bug.\n"
         "Remediation: cordon and drain the bad node, let pods reschedule, then "
         "replace/reboot it.\n"
         "Preventive measure: spread replicas across nodes/zones with topology spread "
         "constraints, set resource requests so the scheduler doesn't overpack nodes, "
         "and alert on node load and NotReady conditions.")
    return q, ctx, a


def disk_pressure(svc, ns):
    q = (f"{svc} pods in {ns} were evicted with reason 'The node was low on resource: "
         f"ephemeral-storage'. Why, and what preventive measure applies?")
    ctx = (f"Events: Evicted — node low on ephemeral-storage (DiskPressure)\n"
           f"$ df -h /var/lib/containerd  ->  use 96%\n"
           f"{svc} writes verbose logs to the container filesystem")
    a = ("Root cause: the node hit DiskPressure (containerd disk 96%) because the app "
         "writes large logs to the container filesystem, so the kubelet evicted pods to "
         "reclaim space.\n"
         "Remediation: clear/rotate the disk, then move logs to stdout (collected by the "
         "node agent) instead of files inside the container.\n"
         "Preventive measure: set ephemeral-storage requests/limits, ship logs off-node, "
         "and alert on node disk usage before it reaches the eviction threshold.")
    return q, ctx, a


def pdb_block(svc, ns):
    q = (f"During a node drain, {svc} in {ns} blocked the operation and one pod stayed "
         f"Terminating. How do PodDisruptionBudgets factor in and what's the right setup?")
    ctx = (f"$ kubectl drain ip-10-2-3-4\n"
           f"error: cannot evict pod {svc}-0: would violate PodDisruptionBudget "
           f"(maxUnavailable=0)\n"
           f"replicas: 1   pdb: minAvailable: 1")
    a = ("Root cause: the PDB requires minAvailable=1 but the Deployment runs a single "
         "replica, so evicting it would violate the budget and the drain is blocked.\n"
         "Remediation: scale to >=2 replicas (or temporarily relax the PDB) so a pod can "
         "be evicted while another stays available.\n"
         "Preventive measure: run >=2 replicas for any drainable workload, set a PDB that "
         "permits one disruption, and add a preStop hook + terminationGracePeriodSeconds "
         "so evictions are graceful.")
    return q, ctx, a


def cpu_throttle(svc, ns):
    q = (f"{svc} in {ns} has high latency but low CPU usage on the dashboard. Could CPU "
         f"limits be throttling it, and how do we prevent that?")
    ctx = (f"container_cpu_cfs_throttled_periods_ratio: 0.62\n"
           f"resources.limits.cpu: 250m   requests.cpu: 100m\n"
           f"p99 latency rose 3x after adding the limit")
    a = ("Root cause: the CPU limit (250m) throttles the container 62% of periods, so "
         "it is CFS-throttled and latency rises even though average CPU looks low.\n"
         "Remediation: raise or remove the CPU limit (keep a realistic request) so the "
         "app can burst.\n"
         "Preventive measure: prefer CPU requests (for scheduling) over tight CPU limits "
         "for latency-sensitive services, and alert on cfs_throttled ratio.")
    return q, ctx, a


def pending(svc, ns):
    q = (f"New {svc} pods in {ns} are stuck Pending and never schedule. What does the "
         f"scheduler event mean and how do we prevent capacity stalls?")
    ctx = (f"$ kubectl -n {ns} describe pod {svc}-abc | tail -2\n"
           f"  0/6 nodes are available: 6 Insufficient cpu.\n"
           f"requests.cpu: 2  per replica; cluster free cpu: 1.3 cores")
    a = ("Root cause: the pod's CPU request (2 cores) exceeds free capacity on every "
         "node (Insufficient cpu), so the scheduler cannot place it and it stays "
         "Pending.\n"
         "Remediation: right-size the request, free capacity, or add nodes / enable "
         "cluster-autoscaler.\n"
         "Preventive measure: set requests from real usage, enable autoscaling with "
         "headroom, and alert on Pending pods so capacity is added before rollouts "
         "stall.")
    return q, ctx, a


def stuck_terminating(svc, ns):
    q = (f"A {svc} pod in {ns} is stuck in Terminating for 10+ minutes during a deploy. "
         f"What causes this and how do we make shutdown clean?")
    ctx = (f"$ kubectl -n {ns} get pod {svc}-9 -> Terminating 12m\n"
           f"terminationGracePeriodSeconds: 30; app ignores SIGTERM\n"
           f"in-flight long requests still draining")
    a = ("Root cause: the app does not handle SIGTERM, so it keeps running until the "
         "grace period forces SIGKILL; long in-flight requests stretch the terminating "
         "state.\n"
         "Remediation: handle SIGTERM to stop accepting new work and finish in-flight "
         "requests; align terminationGracePeriodSeconds with real drain time.\n"
         "Preventive measure: add a preStop hook + readiness flip so the pod leaves "
         "the load balancer before exit, giving zero-downtime, graceful shutdown.")
    return q, ctx, a


def endpoints_empty(svc, ns):
    q = (f"Clients of {svc} in {ns} get connection refused although pods are Running and "
         f"Ready. What's the misconfiguration and the preventive practice?")
    ctx = (f"$ kubectl -n {ns} get endpoints {svc} -> <none>\n"
           f"Service selector: app={svc}-prod\n"
           f"Pod labels: app={svc}")
    a = ("Root cause: the Service selector (app=" + svc + "-prod) does not match the pod "
         "labels (app=" + svc + "), so no endpoints are populated and traffic has nowhere "
         "to go despite healthy pods.\n"
         "Remediation: align the Service selector with the pod template labels.\n"
         "Preventive measure: template labels/selectors from one source (Helm/Kustomize "
         "values) and add a post-deploy check that the Service has >=1 endpoint.")
    return q, ctx, a


def dns_fail(svc, ns):
    q = (f"{svc} in {ns} intermittently fails outbound calls with 'no such host'. Is "
         f"this a DNS problem and how do we prevent it?")
    ctx = (f"app log: dial tcp: lookup db.internal: no such host\n"
           f"$ kubectl -n kube-system get pod -l k8s-app=kube-dns\n"
           f"coredns-xxx  0/1  CrashLoopBackOff   5")
    a = ("Root cause: CoreDNS is unhealthy (CrashLoopBackOff), so name resolution fails "
         "intermittently and the app reports 'no such host' — a cluster DNS problem, not "
         "an app bug.\n"
         "Remediation: fix CoreDNS (often a bad Corefile or resource starvation) and "
         "restart it.\n"
         "Preventive measure: run CoreDNS with >=2 replicas and requests/limits, enable "
         "node-local DNS cache, and alert on CoreDNS health and DNS error rate.")
    return q, ctx, a


def pvc_unbound(svc, ns):
    q = (f"{svc} in {ns} won't start; the pod is Pending on a volume. What does the "
         f"event mean and how do we prevent storage stalls?")
    ctx = (f"Events: pod has unbound immediate PersistentVolumeClaims\n"
           f"PVC {svc}-data  STATUS Pending  storageClass: gp3-fast (not found)")
    a = ("Root cause: the PVC references a StorageClass (gp3-fast) that does not exist, "
         "so it never binds and the pod stays Pending waiting for its volume.\n"
         "Remediation: point the PVC at a valid StorageClass (or create it) so dynamic "
         "provisioning can bind the volume.\n"
         "Preventive measure: validate StorageClass names in CI, set a default "
         "StorageClass, and alert on PVCs Pending longer than a few minutes.")
    return q, ctx, a


BASE_TEMPLATES = [
    ("CrashLoopBackOff", "SEV2", crashloop, "RESTARTS"),
    ("OOMKilled", "SEV2", oomkilled, "exit-137"),
    ("ImagePullBackOff", "SEV3", imagepull, "image-pull"),
    ("ReadinessProbeFailed", "SEV3", readiness, "not-ready"),
    ("LivenessProbeKill", "SEV2", liveness, "restart-loop"),
    ("NodeNotReady", "SEV1", node_notready, "node-lost"),
    ("DiskPressureEviction", "SEV2", disk_pressure, "evicted"),
    ("PDBBlockedDrain", "SEV3", pdb_block, "drain-blocked"),
    ("CPUThrottling", "SEV3", cpu_throttle, "throttled"),
    ("PodPending", "SEV3", pending, "unschedulable"),
    ("StuckTerminating", "SEV3", stuck_terminating, "terminating"),
    ("EndpointsEmpty", "SEV2", endpoints_empty, "no-endpoints"),
    ("DNSFailure", "SEV2", dns_fail, "dns"),
    ("PVCUnbound", "SEV3", pvc_unbound, "volume"),
]


def build_base(target=100):
    """Round-robin the failure modes across services until we reach `target` rows."""
    i = 0
    while len(rows) < target:
        category, sev, fn, signal = BASE_TEMPLATES[i % len(BASE_TEMPLATES)]
        svc, ns = SERVICES[(i // len(BASE_TEMPLATES)) % len(SERVICES)]
        q, ctx, a = fn(svc, ns)
        add(category, sev, q, ctx, a, svc, ns, signal)
        i += 1
        if i > target * 4:  # safety
            break


# ── HERO scenario: node-drain → CrashLoopBackOff on payments-api ──────────────
# This is the question we chat after each run. v1 has light coverage (below);
# v2 adds 50 rows that deepen the root cause + the full set of preventive measures,
# so the fine-tuned answer gets sharper and more complete after retraining.

HERO_Q = ("Our payments-api pods keep entering CrashLoopBackOff right after a node "
          "drain / cluster autoscaler scale-down. What is the most likely root cause, "
          "and what preventive measures stop it from happening again?")

HERO_CTX = ("$ kubectl -n payments get pod\n"
            "payments-api-66d4c8-7q   0/1   CrashLoopBackOff   5   4m   (rescheduled)\n"
            "Events: Back-off restarting failed container; node ip-10-2-7-9 drained\n"
            "$ kubectl -n payments logs payments-api-66d4c8-7q --previous | tail\n"
            "  FATAL: connection pool exhausted: could not acquire DB connection\n"
            "terminationGracePeriodSeconds: 30 | preStop: <none> | replicas: 1 | PDB: <none>")

HERO_A_RICH = (
    "Root cause: when the node is drained, all payments-api replicas are evicted and "
    "rescheduled at once. Because there is no preStop hook and the app ignores SIGTERM, "
    "in-flight DB connections are not released; the pods restart together, stampede the "
    "database, exhaust the connection pool, and crash on boot — a CrashLoopBackOff that "
    "is really a graceful-shutdown + scheduling problem, not application logic.\n"
    "Remediation: scale to >=2 replicas, add a PodDisruptionBudget (maxUnavailable=1), "
    "implement SIGTERM handling with a preStop hook that flips readiness and drains "
    "connections, and add a startupProbe so the slow first boot after reschedule isn't "
    "counted as a crash.\n"
    "Preventive measures: (1) run multiple replicas with topology spread across "
    "nodes/zones; (2) set a PDB so a drain never removes the last healthy pod; "
    "(3) graceful shutdown — handle SIGTERM, preStop sleep, terminationGracePeriodSeconds "
    "longer than real drain time; (4) size DB connection pools and add retry-with-backoff "
    "so a brief reschedule doesn't exhaust them; (5) use startup/readiness probes tuned "
    "to boot time; (6) alert on post-drain restart spikes so the pattern is caught early.")

HERO_A_LIGHT = (
    "Root cause: after a node drain the payments-api pod is rescheduled and restarts; "
    "with a single replica and no graceful shutdown it crashes on boot (CrashLoopBackOff).\n"
    "Remediation: run more than one replica and let it reschedule cleanly.\n"
    "Preventive measure: add a PodDisruptionBudget and handle shutdown so a drain "
    "doesn't take the service down.")


def add_hero_light():
    """Light coverage included in BOTH datasets (a couple of variants)."""
    add("NodeDrainCrashLoop", "SEV1", HERO_Q, HERO_CTX, HERO_A_LIGHT,
        "payments-api", "payments", "drain-crashloop")
    add("NodeDrainCrashLoop", "SEV1",
        "After cluster autoscaler removed a node, checkout-web pods went "
        "CrashLoopBackOff. What's the likely cause and basic preventive step?",
        "Events: node scaled down; checkout-web rescheduled; Back-off restarting\n"
        "replicas: 1 | preStop: <none>",
        "Root cause: a single replica rescheduled after scale-down restarts ungracefully "
        "and crash-loops.\nRemediation: run >=2 replicas.\nPreventive measure: add a PDB "
        "and graceful shutdown so scale-downs are safe.",
        "checkout-web", "storefront", "drain-crashloop")


def build_hero_deep(n=48):
    """v2-only rows: deepen the node-drain CrashLoopBackOff scenario and its
    preventive measures across services and angles, so the retrained model answers
    the hero question with sharper root-cause + a fuller preventive checklist."""
    angles = [
        ("the database connection pool exhausts when all replicas restart together after a drain",
         "size connection pools, add retry-with-backoff, and stagger restarts so a reschedule "
         "doesn't stampede the database"),
        ("there is no PodDisruptionBudget so the drain evicts the only healthy replica",
         "add a PDB with maxUnavailable=1 and run >=2 replicas so a drain never removes the "
         "last healthy pod"),
        ("the app ignores SIGTERM so connections are dropped, not drained, on eviction",
         "handle SIGTERM, add a preStop hook that flips readiness and drains in-flight work, "
         "and set terminationGracePeriodSeconds above real drain time"),
        ("the first boot after reschedule is slow and is mistaken for a crash",
         "add a startupProbe with a generous failureThreshold so slow-but-healthy boots after "
         "a reschedule are not counted as crashes"),
        ("all replicas sit on one node so a single drain takes the whole service down",
         "spread replicas across nodes and zones with topology spread constraints so one drain "
         "cannot evict every replica"),
        ("cluster-autoscaler scale-down is too aggressive during traffic",
         "protect critical pods with the safe-to-evict annotation and PDBs, and set scale-down "
         "delays so drains happen gradually"),
    ]
    out = []
    i = 0
    while len(out) < n:
        svc, ns = SERVICES[i % len(SERVICES)]
        cause, prevent = angles[i % len(angles)]
        inc = f"INC-{4200 + i}"
        q = (f"[{inc}] After a node drain / autoscaler scale-down, {svc} in {ns} entered "
             f"CrashLoopBackOff and we observed that {cause}. What is the root cause and "
             f"the preventive measures?")
        ctx = (f"Incident: {inc}\n"
               f"Events: node drained; {svc} rescheduled; Back-off restarting failed container\n"
               f"replicas: 1 | PDB: <none> | preStop: <none> | grace: 30s\n"
               f"symptom: {cause}")
        a = ("Root cause: a node drain evicts and reschedules the pods at once; because " +
             cause + ", the restart turns into a CrashLoopBackOff. This is a disruption / "
             "graceful-shutdown problem, not an application bug.\n"
             "Remediation: run >=2 replicas and let them reschedule cleanly while one stays "
             "available.\n"
             "Preventive measures: " + prevent + "; combine it with a PDB, multiple replicas, "
             "graceful SIGTERM shutdown, and alerting on post-drain restart spikes so the "
             "pattern is caught before it pages.")
        if (("NodeDrainCrashLoop", q)) not in _seen:
            add("NodeDrainCrashLoop", "SEV1", q, ctx, a, svc, ns, "drain-crashloop")
            out.append(q)
        i += 1
        if i > n * 6:
            break
    # One explicit richly-answered restatement of the hero question so v2 has the
    # full canonical answer to learn from.
    add("NodeDrainCrashLoop", "SEV1",
        "Summarize the complete preventive checklist for payments-api CrashLoopBackOff "
        "after a node drain.",
        HERO_CTX, HERO_A_RICH, "payments-api", "payments", "drain-crashloop")


def write(path, data):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(data)} rows -> {path}")


# ── Build v1 (100 rows): general failure modes + light hero coverage ──────────
build_base(target=98)
add_hero_light()                      # -> 100 rows total
assert len(rows) == 100, f"v1 has {len(rows)} rows"
v1 = list(rows)
write(os.path.join(OUT_DIR, "dataset_v1.jsonl"), v1)

# ── Build v2 (150 rows): v1's 100 + 50 new node-drain/CrashLoop deep rows ──────
build_hero_deep(n=49)                 # +50 (49 angle rows + 1 canonical summary)
assert len(rows) == 150, f"v2 has {len(rows)} rows"
v2 = list(rows)
write(os.path.join(OUT_DIR, "dataset_v2.jsonl"), v2)

# Sanity: v2 is a strict superset of v1 (same first 100 rows).
assert v2[:100] == v1, "v2 must start with v1's 100 rows"
print("v2 is a strict superset of v1 (first 100 rows identical). OK")
print("HERO question:\n  " + HERO_Q)
