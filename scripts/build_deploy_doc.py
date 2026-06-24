#!/usr/bin/env python3
"""Build the self-contained EKS deployment doc (dkubex style) with screenshots + the
demo video embedded as base64. Output: ~/Downloads/finetune-platform-eks-deployment.html
and docs-site/eks-deployment.html (the latter references the video relatively)."""
import base64, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

CHAT = "/tmp/htmlimg/chat.png"
TRAIN = "/tmp/htmlimg/train.png"
VIDEO = os.path.join(ROOT, "demo-appliance.mp4")

CSS = """
:root{--bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;--text:#e6edf3;
--text-secondary:#8b949e;--accent:#58a6ff;--success:#3fb950;--warning:#d29922;--radius:8px;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;}
.container{max-width:900px;margin:0 auto;padding:40px 24px;}
h1{font-size:2rem;margin-bottom:8px;}
.subtitle{color:var(--text-secondary);font-size:1.1rem;margin-bottom:22px;}
h2{font-size:1.3rem;margin:36px 0 12px;color:var(--accent);}
h3{font-size:1.02rem;margin:20px 0 8px;}
p{margin-bottom:12px;color:var(--text-secondary);}
ul,ol{margin:8px 0 16px 22px;color:var(--text-secondary);}
li{margin:6px 0;}
strong{color:var(--text);}
code{background:var(--surface2);padding:2px 6px;border-radius:4px;font-size:.9em;color:var(--accent);}
pre{background:var(--surface2);padding:16px;border-radius:var(--radius);overflow-x:auto;margin:12px 0;border:1px solid var(--border);font-size:13.5px;}
pre code{background:none;padding:0;color:var(--text);}
.badge{display:inline-block;padding:3px 9px;border-radius:4px;font-size:12px;font-weight:500;margin:0 6px 6px 0;}
.badge.blue{background:rgba(88,166,255,.15);color:var(--accent);}
.badge.green{background:rgba(63,185,80,.15);color:var(--success);}
.badge.amber{background:rgba(210,153,34,.15);color:var(--warning);}
.feature-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin:16px 0;}
.feature-card{background:var(--surface2);padding:16px;border-radius:var(--radius);border:1px solid var(--border);}
.feature-card h4{font-size:14px;margin-bottom:6px;}
.feature-card p{font-size:13px;margin:0;}
.step{display:flex;align-items:baseline;gap:10px;margin:26px 0 6px;}
.step .n{flex:0 0 28px;height:28px;line-height:28px;text-align:center;border-radius:50%;background:var(--accent);color:#0d1117;font-weight:700;font-size:14px;}
.step h2{margin:0;}
.note{background:rgba(88,166,255,.08);border-left:3px solid var(--accent);padding:10px 14px;border-radius:4px;margin:14px 0;font-size:14px;color:var(--text-secondary);}
.warn{background:rgba(210,153,34,.08);border-left:3px solid var(--warning);padding:10px 14px;border-radius:4px;margin:14px 0;font-size:14px;color:var(--text-secondary);}
figure{margin:18px 0;}
figure img{width:100%;border-radius:var(--radius);border:1px solid var(--border);display:block;}
figcaption{color:var(--text-secondary);font-size:12.5px;margin-top:8px;text-align:center;}
video{width:100%;border-radius:var(--radius);border:1px solid var(--border);display:block;background:#000;}
hr{border:none;border-top:1px solid var(--border);margin:36px 0;}
.footer{text-align:center;color:var(--text-secondary);font-size:13px;padding:24px 0;}
a{color:var(--accent);text-decoration:none;}a:hover{text-decoration:underline;}
.svg-wrap{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:18px;margin:16px 0;}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px;}
th,td{border:1px solid var(--border);padding:8px 10px;text-align:left;}
th{background:var(--surface2);color:var(--text);}td{color:var(--text-secondary);}
"""

SVG = """
<svg viewBox="0 0 760 280" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,sans-serif">
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#8b949e"/></marker></defs>
  <text x="18" y="38" fill="#e6edf3" font-size="13" font-weight="600">User</text>
  <text x="18" y="54" fill="#8b949e" font-size="11">browser</text>
  <line x1="18" y1="66" x2="18" y2="150" stroke="#8b949e" stroke-width="1.3" marker-end="url(#ar)"/>
  <text x="26" y="112" fill="#8b949e" font-size="11">LoadBalancer / port-forward</text>
  <rect x="170" y="18" width="572" height="246" rx="10" fill="none" stroke="#30363d" stroke-dasharray="5 4"/>
  <text x="186" y="38" fill="#58a6ff" font-size="12" font-weight="600">Kubernetes (one stateful pod)</text>
  <rect x="190" y="58" width="350" height="150" rx="8" fill="#1c2128" stroke="#30363d"/>
  <text x="204" y="80" fill="#58a6ff" font-size="12" font-weight="600">finetune-platform</text>
  <rect x="206" y="92" width="150" height="34" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="216" y="113" fill="#e6edf3" font-size="11" font-weight="600">UI :7100 / infer :7200</text>
  <rect x="366" y="92" width="158" height="34" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="376" y="113" fill="#3fb950" font-size="11" font-weight="600">ollama :11434 (baked)</text>
  <text x="206" y="150" fill="#8b949e" font-size="11">baked: HF models · sre-assistant · 43 SRE tables</text>
  <text x="206" y="168" fill="#8b949e" font-size="11">serve.sh seeds data on first boot</text>
  <text x="206" y="186" fill="#8b949e" font-size="11">cards · text-to-SQL · train/retrain · upload</text>
  <line x1="18" y1="150" x2="190" y2="120" stroke="#8b949e" stroke-width="1.3" marker-end="url(#ar)"/>
  <rect x="566" y="58" width="158" height="150" rx="8" fill="#1c2128" stroke="#30363d"/>
  <text x="580" y="80" fill="#58a6ff" font-size="12" font-weight="600">EBS PVCs (optional)</text>
  <text x="580" y="104" fill="#8b949e" font-size="11">data · models · logs</text>
  <text x="580" y="128" fill="#8b949e" font-size="11">(for appended data</text>
  <text x="580" y="144" fill="#8b949e" font-size="11">+ retrained models)</text>
  <line x1="540" y1="150" x2="566" y2="150" stroke="#8b949e" stroke-width="1.3" marker-end="url(#ar)"/>
</svg>
"""

def build(video_tag):
    chat_b64 = b64(CHAT); train_b64 = b64(TRAIN)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Deploy the Fine-Tuning Appliance on Kubernetes</title><style>{CSS}</style></head>
<body><div class="container">

<h1>🚀 Fine-Tuning Appliance — Deploy on Kubernetes</h1>
<p class="subtitle">A self-contained SRE fine-tuning app: bundled data, base models, and a
pre-trained model — installed with one Helm command, runs fully offline.</p>
<div>
  <span class="badge green">Self-contained</span>
  <span class="badge blue">Air-gapped</span>
  <span class="badge blue">amd64 / CPU</span>
  <span class="badge amber">2 commands</span>
</div>

<h2>What you get</h2>
<div class="feature-grid">
  <div class="feature-card"><h4><span class="badge green">Baked</span> Everything in the image</h4><p>Base models (Qwen2.5 0.5B + 1.5B), 43 SRE observability tables, training data, and a pre-trained <code>sre-assistant</code> — no downloads.</p></div>
  <div class="feature-card"><h4><span class="badge blue">Cards</span> Query your data</h4><p>One-tap preset cards (pods at risk, OOM offenders…) and a "Query data" mode that writes SQL from plain English. <strong>Add your own card</strong> with a title + <code>SELECT</code> — no rebuild.</p></div>
  <div class="feature-card"><h4><span class="badge green">Tune</span> Train &amp; retrain</h4><p>Fine-tune on bundled data, then <em>Continue from fine-tuned</em> on more data — the answers measurably improve.</p></div>
  <div class="feature-card"><h4><span class="badge blue">Yours</span> Bring your own data</h4><p>Two ways, no code: <strong>⬆ Data table</strong> uploads a CSV as a queryable table (cards + text-to-SQL see it); <strong>⬆ Upload CSV/JSONL</strong> adds training data to the dropdown, ready to retrain on.</p></div>
</div>

<figure><img src="data:image/png;base64,{chat_b64}" alt="Chat with cards + a query result"><figcaption>The chat — preset cards and text-to-SQL answer straight from the bundled data.</figcaption></figure>

<h2>How it runs</h2>
<p>One stateful pod. The base models, data, and adapter are baked into the image; an
optional Ollama sidecar (also baked) provides base-model chat. Nothing is downloaded at
runtime — the only network use is pulling the two public images once.</p>
<div class="svg-wrap">{SVG}</div>

<div class="step"><span class="n">1</span><h2>Install — one command</h2></div>
<p>Point <code>kubectl</code> at your cluster (with a <code>gp3</code> StorageClass) and install
straight from the registry — <strong>no <code>git clone</code>, no source to download</strong>:</p>
<pre><code>helm install finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 \\
  --namespace finetune --create-namespace \\
  --set persistence.storageClass=gp3</code></pre>
<p>The app + baked Ollama images (amd64) are the chart defaults — nothing else to set.
(Want to customise? <code>git clone</code> the repo and install from <code>charts/finetune-platform</code> instead.)</p>
<div class="note"><strong>Brand-new to AWS?</strong> The <code>AWS-CONSOLE-GUIDE.md</code> has a
full click-by-click version using AWS CloudShell (no software on your computer) — log in,
open CloudShell, paste a few commands, open a link. Use a <strong>m5.2xlarge</strong> node.</div>

<div class="step"><span class="n">2</span><h2>Open it</h2></div>
<pre><code>kubectl -n finetune rollout status deploy/finetune-platform   # first boot pulls the images
kubectl -n finetune port-forward svc/finetune-platform 7100:7100
# open http://localhost:7100</code></pre>
<p>Or expose a URL with <code>--set service.type=LoadBalancer</code>.</p>

<h2>Explore the data (Chat tab)</h2>
<ul>
  <li><strong>Tap a card</strong> — e.g. <em>Top OOM offenders</em> or <em>CrashLoopBackOff pods</em> → a live table from the bundled SRE data appears.</li>
  <li>Tick <strong>🗄️ Query data</strong>, ask in plain English: <code>top 5 namespaces by alert count</code> → it writes the SQL and shows the answer.</li>
  <li>Pick <strong>sre-assistant (fine-tuned)</strong> in the model dropdown → ask an SRE question → a grounded remediation answer.</li>
</ul>

<h2>Make it yours — no code, no rebuild</h2>
<p>Both are buttons on the Chat tab, next to the cards. Nothing to redeploy.</p>
<ul>
  <li><strong>➕ Card</strong> → give it a <em>title</em> and a <code>SELECT …</code> query → <em>Save</em>. It's validated (read-only only), saved to <code>data/cards.json</code> on the volume, and shows up in the strip for everyone. Hit <strong>✕</strong> on a custom card to remove it.</li>
  <li><strong>⬆ Data table</strong> → upload a CSV of your <em>own</em> observability data → name the table → it becomes a queryable DuckDB table. Your cards <em>and</em> the 🗄️ text-to-SQL chat can now query it immediately.</li>
</ul>

<h2>See training improve — the v1 → v2 demo</h2>
<p>This is the core loop: train once, ask a hard question, then <em>continue from the fine-tuned
model</em> on more data and watch the same answer get sharper. Two datasets ship for exactly this —
<code>sre_qa_v1</code> (141 rows) and <code>sre_qa_v2</code> (241 rows, with the harder OOM/remediation cases).</p>
<ol>
  <li><strong>Train v1.</strong> Train tab → dataset <code>sre_qa_v1</code> → <strong>Start Training</strong> → watch the loss curve settle → <strong>Export &amp; Serve</strong>. The new model lands in the Chat model dropdown.</li>
  <li><strong>Ask the hero question.</strong> Chat tab → select the v1 model → ask:
    <br><code>Our ledger-svc pod in finance was flagged for OOM Risk at high risk. Give the full root cause and remediation.</code>
    <br>v1 answers, but generically — it hasn't seen many OOM-remediation examples yet.</li>
  <li><strong>Retrain into v2.</strong> Train tab → dataset <code>sre_qa_v2</code> → tick <strong>Continue from fine-tuned</strong> and pick your v1 model → <strong>Start Training</strong>. The loss curve starts <em>lower</em> than v1 did (it's building on v1, not from scratch) and settles lower. → <strong>Export &amp; Serve</strong>.</li>
  <li><strong>Ask again.</strong> Select the v2 model, ask the <em>same</em> question → the answer is now specific: OOMKill (exit 137) cause, the memory-limit fix, and the remediation steps. That delta is the demo.</li>
</ol>
<div class="note"><strong>Want to train on your own Q&amp;A?</strong> Train tab → <strong>⬆ Upload CSV / JSONL</strong>.
CSV columns <code>question,reference_answer[,context]</code> (or <code>prompt,completion</code>) — it converts to
JSONL, lands in the dropdown, and you train/retrain on it exactly like v1/v2.</div>
<div class="warn"><strong>Persistence:</strong> custom cards, uploaded tables, and retrained models live on the
PVCs — install with <code>--set persistence.storageClass=gp3</code> (the default) so they survive restarts.
With <code>persistence.enabled=false</code> they're ephemeral (fine for a quick demo).</div>
<figure><img src="data:image/png;base64,{train_b64}" alt="Train tab — dataset dropdown, upload, loss curve"><figcaption>The Train tab — dataset dropdown, upload-your-own-data, Continue-from-fine-tuned, and the live loss curve.</figcaption></figure>

<h2>Air-gapped clusters</h2>
<p>The app needs no internet at runtime — only to pull two public images once. Mirror them
into your private registry and set <code>--set image.repository=&lt;your-registry&gt;/finetune-platform</code>
(and <code>ollama.image</code>); then it runs completely offline.</p>

<h2>Watch it in action</h2>
<p>A ~3-minute walkthrough — bundled data, the dataset dropdown, two training trials, and
the answer getting better on the second pass (no audio):</p>
{video_tag}

<hr>
<div class="footer">Fine-Tuning Appliance · single pod · amd64 / CPU · Helm chart at <code>charts/finetune-platform</code> · github.com/t4tarzan/finetune-platform</div>
</div></body></html>"""

# 1) self-contained version for Downloads (video embedded)
vid_b64 = b64(VIDEO)
embedded_video = f'<video controls preload="metadata" poster="data:image/png;base64,{b64(CHAT)}"><source src="data:video/mp4;base64,{vid_b64}" type="video/mp4"></video>'
out1 = os.path.expanduser("~/Downloads/finetune-platform-eks-deployment.html")
open(out1, "w").write(build(embedded_video))
print(f"wrote {out1} ({os.path.getsize(out1)//1024//1024} MB, self-contained w/ video)")

# 2) repo docs-site version: reference the video relatively (kept light)
ref_video = '<video controls preload="metadata"><source src="demo-appliance.mp4" type="video/mp4"></video>'
out2 = os.path.join(ROOT, "docs-site", "eks-deployment.html")
open(out2, "w").write(build(ref_video))
print(f"wrote {out2} ({os.path.getsize(out2)//1024} KB, references demo-appliance.mp4)")
