# Install Fine-Tuning Platform on a fresh EKS cluster

This kit installs the platform as a **single stateful pod** (web UI `:7100` +
OpenAI-compatible inference `:7200`) on Amazon EKS, using the **HuggingFace/PyTorch
CPU backend** (works on any amd64 node — e.g. `m5.xlarge`; MLX/Metal is Mac-only and
not used in-cluster).

You build the image from the included source and push it to **your own registry**
(ECR or GHCR), then `helm install` the included chart. No dependency on any private
image.

---

## 0. Prerequisites

On your workstation:
- `kubectl` (pointed at your EKS cluster — `aws eks update-kubeconfig --name <cluster> --region <region>`)
- `helm` v3
- `docker` with **buildx** (to build a linux/amd64 image)
- `aws` CLI (only if pushing to ECR)

On the cluster:
- **EBS CSI driver** + a StorageClass (the chart defaults to `gp3`). Check: `kubectl get storageclass`
- An **amd64 node group** with room for the model (≥ 8 GB RAM for 0.5B/1.5B; more for 7B).
- *(Optional)* an ingress controller (ALB or ingress-nginx) if you want a URL instead of `port-forward`.

Verify:
```bash
kubectl get nodes -o wide          # confirm amd64 + Ready
kubectl get storageclass           # confirm a default / gp3 exists
```

---

## 1. Build and push the image (linux/amd64)

### Option A — Amazon ECR (recommended on EKS)
```bash
export AWS_REGION=us-east-2
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REG=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
export IMAGE=$REG/finetune-platform

aws ecr create-repository --repository-name finetune-platform --region $AWS_REGION 2>/dev/null || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REG

# EKS nodes are amd64 — build for that platform explicitly (esp. on an Apple Mac)
docker buildx build --platform linux/amd64 -t $IMAGE:latest --push .
```
ECR repos are private but the EKS node role can pull them by default (no pull secret
needed) — make sure the node IAM role has `AmazonEC2ContainerRegistryReadOnly`.

### Option B — GitHub Container Registry (GHCR)
```bash
export IMAGE=ghcr.io/<your-user-or-org>/finetune-platform
echo $GHCR_TOKEN | docker login ghcr.io -u <your-user> --password-stdin   # token needs write:packages
docker buildx build --platform linux/amd64 -t $IMAGE:latest --push .
```
If the GHCR package is **private**, create a pull secret and pass it in step 2:
```bash
kubectl -n finetune create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io --docker-username=<user> --docker-password=$GHCR_TOKEN
# then add:  --set 'imagePullSecrets[0].name=ghcr-pull'
```
(Make the package public instead and you can skip the secret.)

---

## 2. Install with Helm

```bash
helm install finetune-platform charts/finetune-platform \
  --namespace finetune --create-namespace \
  --set image.repository=$IMAGE \
  --set image.tag=latest \
  --set persistence.storageClass=gp3 \
  --set ollama.enabled=true          # bundled base-model chat sidecar
  # add for a private GHCR image:  --set 'imagePullSecrets[0].name=ghcr-pull'
```

Wait for it (first boot downloads the base model into the HF-cache PVC — a few minutes):
```bash
kubectl -n finetune rollout status deploy/finetune-platform
```

The chart creates: the Deployment (1 replica, `Recreate`), a Service (7100/7200),
four EBS PVCs (`models`, `data`, `hf-cache`, `logs`), and an init container that
**seeds the two demo datasets** into the data volume so they appear in the Train
dropdown.

---

## 3. Open the UI

**Quick (no ingress):**
```bash
kubectl -n finetune port-forward svc/finetune-platform 7100:7100
# open http://localhost:7100
```

**LoadBalancer:**
```bash
helm upgrade finetune-platform charts/finetune-platform -n finetune --reuse-values \
  --set service.type=LoadBalancer
kubectl -n finetune get svc finetune-platform -w        # grab EXTERNAL-IP
```

**Ingress — own host (ALB example):**
```bash
helm upgrade finetune-platform charts/finetune-platform -n finetune --reuse-values \
  --set ingress.enabled=true --set ingress.className=alb \
  --set ingress.host=finetune.example.com
```

**Ingress — shared nginx under a sub-path** (e.g. `https://<host>/finetune-platform`):
```bash
helm upgrade finetune-platform charts/finetune-platform -n finetune --reuse-values \
  --set ingress.enabled=true --set ingress.className=nginx \
  --set basePath=/finetune-platform
```
> `basePath` makes the app serve under that prefix: it sets `BASE_PATH` in the pod
> (so the UI prefixes every API call) and auto-configures the nginx ingress (regex
> path + `rewrite-target` that strips the prefix). Leave `basePath` empty for a
> dedicated host or LoadBalancer.

---

## 4. Pull a base model (for base-model chat)

The Ollama sidecar starts empty. Pull one so the model dropdown isn't empty:
```bash
kubectl -n finetune exec -c ollama deploy/finetune-platform -- ollama pull qwen2.5:0.5b
```
(Fine-tuned models you train+export don't need this — they're served by `:7200`.)

---

## 5. Run the demo

In the UI's **Train** tab:
1. **Dataset** dropdown → `data/sre-pods/dataset_v1.jsonl` (100 rows) → **Use This Dataset**
2. Base model `Qwen2.5-1.5B (4-bit)`, rows `120` → **Start Training** (watch the loss curve)
3. **⬆ Export & Serve** → go to **Chat**, pick the fine-tuned model, ask a question
4. Back in **Train**: dataset `dataset_v2.jsonl` (150 rows), **Continue from fine-tuned →**
   the v1 model, rows `200` → **Start Training** (loss continues lower)
5. **Export & Serve** → **Chat** again → the answer is sharper

> CPU training is slow — keep to **0.5B/1.5B + small datasets** on a CPU node. For
> real speed, use a GPU node group + a CUDA image (see `charts/finetune-platform/README.md`).

---

## Uninstall
```bash
helm uninstall finetune-platform -n finetune
kubectl -n finetune delete pvc -l app.kubernetes.io/name=finetune-platform   # also drops data
```

## Notes
- **Single replica only** — the app is stateful (in-memory training state + RWO
  volumes). Scale the node, not the pod count.
- Sizing: `resources` in `values.yaml` default to 1–3 CPU / 4–12 GiB. Raise for
  larger models.
- Full chart options: `charts/finetune-platform/README.md`.
