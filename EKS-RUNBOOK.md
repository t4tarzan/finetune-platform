# Deploy the Fine-Tuning Appliance on EKS — Step-by-Step Runbook

Follow these in order. End result: the app running on your EKS cluster with bundled
data, preset cards, text-to-SQL chat, and a pre-trained `sre-assistant` — from one
`helm install`, no internet needed at runtime beyond pulling the image.

> The image is **amd64** and **self-contained** (base models + 43 SRE tables + a
> pre-trained adapter all baked in). You only pull it and `helm install`.

---

## 0. Install these tools (once, on your laptop)
- [ ] `aws` CLI — configured (`aws configure`) with access to the account
- [ ] `kubectl`
- [ ] `helm` (v3.8+ — for OCI registry installs)
- [ ] `eksctl` — only if you need to **create** a cluster in step 1
- [ ] `git` — **optional**, only if you want to build/customise from source instead

## 1. Get a cluster + point kubectl at it
- **Option A — you already have an EKS cluster:**
  ```bash
  aws eks update-kubeconfig --name <CLUSTER_NAME> --region <REGION>
  ```
- **Option B — create a fresh one (~15 min, amd64 node):**
  ```bash
  eksctl create cluster \
    --name finetune-test --region us-east-2 \
    --node-type m5.2xlarge --nodes 1 --managed
  ```
  - `m5.2xlarge` (8 vCPU / 32 GB) is comfortable for 0.5B/1.5B on CPU.
    `m5.xlarge` (16 GB) is the minimum.
- **Verify:**
  ```bash
  kubectl get nodes -o wide        # 1 node, STATUS Ready, amd64
  ```

## 2. Confirm storage (EBS CSI + a gp3 StorageClass)
- [ ] Check:
  ```bash
  kubectl get storageclass         # look for one named gp3 (or a default)
  ```
- If there is **no gp3** (eksctl clusters usually have `gp2` only):
  ```bash
  # enable the EBS CSI driver addon (one-time)
  eksctl create addon --name aws-ebs-csi-driver --cluster finetune-test --region us-east-2 --force
  # create a gp3 StorageClass
  kubectl apply -f - <<'EOF'
  apiVersion: storage.k8s.io/v1
  kind: StorageClass
  metadata: { name: gp3 }
  provisioner: ebs.csi.aws.com
  parameters: { type: gp3 }
  volumeBindingMode: WaitForFirstConsumer
  allowVolumeExpansion: true
  EOF
  ```
  - Or just use your existing class and pass its name in step 3.

## 3. Deploy — one command (no download)
The chart is published to the registry, so install it directly — **no `git clone`, no
source to fetch**:
```bash
helm install finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 \
  --namespace finetune --create-namespace \
  --set persistence.storageClass=gp3
```
- The app + Ollama images (amd64) are the chart defaults. Base-model chat (air-gapped,
  baked models) is **on by default** — needs a `m5.2xlarge`-sized node. Drop it with
  `--set ollama.enabled=false`.
- Different StorageClass? change `--set persistence.storageClass=<name>`.
- *Prefer to customise / build your own?* `git clone` the repo and
  `helm install finetune-platform charts/finetune-platform …` from the local path instead.

## 4. Wait for it to come up
```bash
kubectl -n finetune rollout status deploy/finetune-platform
```
- First boot pulls the image (~3.7 GB) and **auto-seeds** the data + merges the
  `sre-assistant` adapter (no action needed). Give it a few minutes.
- Watch logs if you like:
  ```bash
  kubectl -n finetune logs deploy/finetune-platform -f
  # expect: "[serve] seeded bundled datasets + adapter…" and "[startup] seeded 43 observability tables…"
  ```

## 5. Open the UI
```bash
kubectl -n finetune port-forward svc/finetune-platform 7100:7100
# then open http://localhost:7100
```

## 6. Verify the appliance (this is the test)
- [ ] **Chat tab** → tap a card (e.g. **Top OOM offenders**) → a table from the bundled data appears.
- [ ] Tick **🗄️ Query data**, ask `top 5 namespaces by alert count` → it shows the SQL + the answer.
- [ ] **Model dropdown** lists **`sre-assistant (fine-tuned)`** → pick it, ask
  *"a payments pod was flagged for OOM Risk — what do I do?"* → a remediation answer.
- [ ] **Train tab** → pick `dataset_v1.jsonl` → **Start Training** → **Export & Serve** →
  then `dataset_v2.jsonl` with **Continue from fine-tuned** → retrain → chat again (improves).
- [ ] **Bring your own data**: Train tab → **⬆ Upload CSV / JSONL** → pick a file
  (CSV columns `question,reference_answer[,context]` or `prompt,completion`) → it
  converts, lands in the dropdown, and you can train/retrain on it.
  > For uploads + retrained models to **persist** across restarts, install with the
  > PVC path (`--set persistence.storageClass=gp3`, the default). With
  > `persistence.enabled=false` they're ephemeral (fine for a demo).

## 7. (Optional) Expose without port-forward
- **LoadBalancer** (gets an external address):
  ```bash
  helm upgrade finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 -n finetune --reuse-values \
    --set service.type=LoadBalancer
  kubectl -n finetune get svc finetune-platform -w     # copy EXTERNAL-IP
  ```
- **nginx ingress under a sub-path** (`https://<host>/finetune-platform`):
  ```bash
  helm upgrade finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 -n finetune --reuse-values \
    --set ingress.enabled=true --set ingress.className=nginx --set basePath=/finetune-platform
  ```

## 7b. Update an existing install to the latest image
The published image (`:latest`) moves forward as the app improves. A node caches the old
`:latest`, so a plain `helm upgrade` won't re-pull — force it:
```bash
helm upgrade finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 -n finetune --reuse-values \
  --set image.pullPolicy=Always
kubectl -n finetune rollout restart deploy/finetune-platform
kubectl -n finetune rollout status deploy/finetune-platform
```
On restart, `serve.sh` re-seeds **only missing** bundled files (`cp -n`) — your uploads
and retrained models on the PVC are untouched.

## 8. Tear down (avoid AWS charges)
```bash
helm uninstall finetune-platform -n finetune
kubectl -n finetune delete pvc -l app.kubernetes.io/name=finetune-platform   # deletes the data
# if you created the cluster in step 1B:
eksctl delete cluster --name finetune-test --region us-east-2
```

---

### Air-gapped clients
Mirror the **two images** (app + ollama) into your private registry, then in step 3 add
`--set image.repository=<your-registry>/finetune-platform` and
`--set ollama.image=<your-registry>/finetune-ollama:latest`. Nothing else changes —
data, models, and the adapter are all inside the images.
