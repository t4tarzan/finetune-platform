# finetune-platform Helm chart

Deploys the Fine-Tuning Platform as a **single stateful pod** (web UI `:7100` +
OpenAI-compatible inference `:7200`) on Kubernetes / EKS.

In-cluster it runs the **HuggingFace/PyTorch backend** (containers are Linux, so the
Apple-Silicon MLX/Metal path is not used). On a CPU node it works but is slow — size
models to the node (0.5B / 1.5B + small datasets on something like an `m5.xlarge`;
GPU node group for real throughput, see below).

## 1. Build and push the image

The cluster needs to pull the image from a registry it can reach (ECR/GHCR/…):

```bash
# from the repo root
aws ecr create-repository --repository-name finetune-platform 2>/dev/null || true
REG=123456789012.dkr.ecr.us-east-1.amazonaws.com         # your registry
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REG"

# EKS nodes are amd64 — build for that platform explicitly if you're on an Apple Mac
docker buildx build --platform linux/amd64 -t "$REG/finetune-platform:latest" --push .
```

## 2. Install

```bash
helm install ftp charts/finetune-platform \
  --namespace finetune --create-namespace \
  --set image.repository=$REG/finetune-platform \
  --set image.tag=latest \
  --set persistence.storageClass=gp3        # your EBS StorageClass; "" = cluster default
```

First boot downloads the base model from Hugging Face (into the `hfcache` PVC) — give
the rollout a few minutes:

```bash
kubectl -n finetune rollout status deploy/ftp-finetune-platform
kubectl -n finetune port-forward svc/ftp-finetune-platform 7100:7100
open http://localhost:7100
```

Reproduce the SRE demo inside the pod:

```bash
kubectl -n finetune exec deploy/ftp-finetune-platform -- bash scripts/run_sre_demo.sh
```

## Storage

Four `ReadWriteOnce` PVCs (that's why the pod is single-replica):

| PVC | mount | default | holds |
|-----|-------|---------|-------|
| `…-models`  | `/app/models` | 20Gi | LoRA adapters + merged models (GB each) |
| `…-data`    | `/app/data`   | 5Gi  | datasets + training history |
| `…-hfcache` | `/root/.cache/huggingface` | 30Gi | base-model downloads |
| `…-logs`    | `/app/logs`   | 1Gi  | training/export/server logs |

Need >1 pod to share these? Switch to an **EFS** (RWX) StorageClass.

## Base-model chat (optional Ollama sidecar)

Fine-tuned models are served by `:7200` and work without Ollama. For *base*-model
chat, enable the sidecar and pull a model:

```bash
helm upgrade ftp charts/finetune-platform ... --set ollama.enabled=true
kubectl -n finetune exec -c ollama deploy/ftp-finetune-platform -- ollama pull qwen2.5:0.5b
```

## Expose the UI

- `--set service.type=LoadBalancer`, or
- `--set ingress.enabled=true --set ingress.host=finetune.example.com`
  (defaults to the AWS Load Balancer Controller, `ingressClassName: alb`; switch
  `ingress.className`/annotations for ingress-nginx).

## GPU (real training throughput)

The bundled image is **CPU-only**. For GPU:
1. Build a CUDA image (swap the base + install a CUDA torch wheel) and push it.
2. Run on a GPU node group (e.g. `g5`) with the NVIDIA device plugin installed.
3. Add a GPU request/limit and pin to the GPU pool, e.g.:
   ```bash
   helm install ftp charts/finetune-platform ... \
     --set-json 'resources.limits={"nvidia.com/gpu":1,"cpu":"4","memory":"24Gi"}' \
     --set nodeSelector."eks\.amazonaws\.com/nodegroup"=gpu-pool
   ```

## Common overrides

| key | default | note |
|-----|---------|------|
| `image.repository` | `""` (**required**) | your registry image |
| `image.tag` | `latest` | |
| `persistence.storageClass` | `gp3` | `""` = cluster default |
| `persistence.models.size` | `20Gi` | raise for many/large models |
| `resources.requests/limits` | 1–3 CPU / 4–12Gi | size to the node + model |
| `service.type` | `ClusterIP` | or `LoadBalancer` |
| `ingress.enabled` | `false` | UI ingress (`:7100`) |
| `basePath` | `""` | serve under a URL sub-path (e.g. `/finetune-platform`) on shared nginx — sets `BASE_PATH` + auto rewrite |
| `ollama.enabled` | `false` | base-model chat sidecar |

Uninstall: `helm uninstall ftp -n finetune` (PVCs remain — delete them by hand to drop data).
