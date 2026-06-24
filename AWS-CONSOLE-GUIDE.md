# Deploy the Fine-Tuning Appliance on AWS — Click-by-Click (no software to install)

For someone who just has an **AWS account they can log into**. You'll do everything in
the **AWS web console + its built-in terminal (CloudShell)** — nothing is installed on
your own computer. Total time ≈ **35–45 min** (most of it is waiting for AWS).

> 💵 **Cost:** while it's running it's roughly **$0.50–0.60 per hour** (cluster + 1
> server + load balancer). **Step 8 deletes everything** — do that when you're done.

---

1. **Log in to AWS**
   1. Open **https://console.aws.amazon.com** in your browser and sign in.
   2. **Top-right corner**, click the **Region** name (e.g. *"N. Virginia"*).
      - Pick **US East (N. Virginia) us-east-1** (any region is fine — just remember which).

2. **Open CloudShell** (the free terminal built into AWS — no install)
   1. In the **top navigation bar**, click the **`>_`** icon (tooltip says *"CloudShell"*).
      - If you can't see it: use the search box at the top, type **CloudShell**, click it.
   2. A black terminal opens at the bottom. Wait ~30 seconds for the prompt to appear.
   3. **How to paste:** copy a command block from this guide, then in CloudShell press
      **Ctrl + V** (Windows) / **Cmd + V** (Mac), then press **Enter**.
      - If paste is blocked, **right-click → Paste**.

3. **Install the 3 helper tools** (paste each block, press Enter, wait for it to finish)
   1. **eksctl** (creates the cluster):
      ```bash
      curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" | tar xz -C /tmp
      sudo mv /tmp/eksctl /usr/local/bin && eksctl version
      ```
   2. **kubectl** (talks to the cluster):
      ```bash
      curl -sLO "https://dl.k8s.io/release/v1.30.4/bin/linux/amd64/kubectl"
      chmod +x kubectl && sudo mv kubectl /usr/local/bin/ && kubectl version --client
      ```
   3. **helm** (installs the app):
      ```bash
      curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && helm version
      ```
   - ✅ Each should print a version line with no red errors.

4. **Create the Kubernetes cluster** (one command, then **~18 minutes of waiting**)
   1. Set your region (change `us-east-1` if you picked another in step 1):
      ```bash
      export REGION=us-east-1
      ```
   2. Create it:
      ```bash
      eksctl create cluster --name finetune-test --region $REGION \
        --node-type m5.2xlarge --nodes 1 --managed
      ```
   3. It will print **many lines** for ~15–20 min. ☕ Wait until you see a line like
      **`EKS cluster "finetune-test" in "..." region is ready`**.
   4. Check it worked:
      ```bash
      kubectl get nodes
      ```
      - ✅ Expect **one** line with **`STATUS Ready`**.

5. **Install the app — one command** (nothing to download first)
   - The app is published to a registry, so this single command installs it (no source
     code, no `git`). It needs **no storage setup** (data is inside the image) and gives
     you a **public web link**:
     ```bash
     helm install finetune-platform oci://ghcr.io/t4tarzan/charts/finetune-platform --version 0.1.0 \
       --namespace finetune --create-namespace \
       --set persistence.enabled=false \
       --set service.type=LoadBalancer
     ```
   - ✅ Ends with **`STATUS: deployed`**.

6. **Wait for it to start, then get your link**
   1. Wait for the app to be ready (it downloads a ~3.7 GB image the first time):
      ```bash
      kubectl -n finetune rollout status deploy/finetune-platform
      ```
      - ✅ Ends with **`successfully rolled out`** (give it 3–6 min).
   2. Get your web address:
      ```bash
      kubectl -n finetune get svc finetune-platform
      ```
      - Look at the **`EXTERNAL-IP`** column — it's a long AWS address like
        `a1b2c3...elb.amazonaws.com`.
      - If it says **`<pending>`**, wait 2 minutes and run the command again until the
        address appears.
   3. **Your link is:**  `http://<EXTERNAL-IP>:7100`
      - Example: `http://a1b2c3d4e5.us-east-1.elb.amazonaws.com:7100`
      - (Don't forget the **`:7100`** at the end.)

7. **Open it in your browser and try it**
   1. Paste the `http://<EXTERNAL-IP>:7100` link into a new browser tab.
      - First open can take ~1 min while it warms up. If it doesn't load, wait and refresh.
   2. Things to try:
      1. **Chat tab** → click a card like **"Top OOM offenders"** → a table of data appears.
      2. Tick **🗄️ Query data**, type `top 5 namespaces by alert count`, press **Send** →
         it writes the query and shows the answer.
      3. In the **model dropdown** (top-left), pick **`sre-assistant (fine-tuned)`** and
         ask *"a payments pod was flagged for OOM risk — what do I do?"*.
      4. **Train tab** → **Dataset** dropdown → pick `sre_qa_v1.jsonl` → **Use This Dataset**
         → **Start Training** → watch the loss curve → **Export & Serve** → then pick
         `sre_qa_v2.jsonl`, set **Continue from fine-tuned**, retrain → the answer improves.
      5. **Use your own data** → Train tab → **⬆ Upload CSV / JSONL** → choose a file (a
         CSV with columns `question,reference_answer` works) → it's added to the dropdown
         → train/retrain on it.
         - Note: in *this* simple setup the data is **not saved** if the server restarts
           (we skipped storage to keep it easy). For data that sticks around, use the
           `EKS-RUNBOOK.md` install instead.

8. **🧹 DELETE EVERYTHING when done** (so AWS stops charging you)
   1. In CloudShell:
      ```bash
      helm uninstall finetune-platform -n finetune
      eksctl delete cluster --name finetune-test --region $REGION
      ```
      - The delete takes ~10 min and prints **`all cluster resources were deleted`**.
   2. Double-check in the console: top search box → **EKS** → **Clusters** → the list
      should be **empty** (or not show `finetune-test`).
   3. Also check **EC2 → Load Balancers** is empty (the app's load balancer is gone).

---

## If something goes wrong
- **`EXTERNAL-IP` stays `<pending>` forever** → your account may not auto-create load
  balancers. Easiest fix: use a tunnel instead — in CloudShell run
  `kubectl -n finetune port-forward svc/finetune-platform 7100:7100` and use CloudShell's
  **Actions → "Port forwarding"** (or just confirm the app is healthy with
  `kubectl -n finetune get pods` → `READY 1/1`).
- **Pod stuck `Pending`** → the node is too small; recreate the cluster in step 4 with
  `--node-type m5.4xlarge`.
- **Anything red in step 3** → re-run that one block; CloudShell sometimes needs a retry.
- **Forgot to delete and worried about cost** → just run the two commands in step 8 — that
  removes all the paid resources.

## Air-gapped / private clusters
This pulls one public image (`ghcr.io/t4tarzan/finetune-platform:latest`). For a closed
network, mirror that image into your private registry and add
`--set image.repository=<your-registry>/finetune-platform` to the step-5 command.
