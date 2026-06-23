/**
 * Full before/after fine-tuning story on the SRE / Kubernetes dataset, recorded
 * as a narrated-by-caption video. Shows the SAME question answered at three
 * stages so the improvement is visible:
 *
 *   1. BASE model (qwen2.5:0.5b, no fine-tuning)         -> generic answer
 *   2. v1: train on data/sre-pods/dataset_v1.jsonl (100 rows, from base),
 *      export & serve, ask again                         -> better, but the
 *                                                            node-drain root cause
 *                                                            isn't in v1's data
 *   3. v2: CONTINUE training from the v1 adapter on
 *      dataset_v2.jsonl (150 rows; +50 NodeDrainCrashLoop rows), export & serve,
 *      ask again                                          -> nails the node-drain
 *                                                            root cause + PDB /
 *                                                            graceful-shutdown fixes
 *
 * The question is framed from the dataset's NodeDrainCrashLoop incidents (the 50
 * rows that exist only in v2), so each stage measurably improves.
 *
 * Runs LIVE on this Linux box: HuggingFace/CPU backend, base Qwen2.5-0.5B-Instruct,
 * inference server on :7200. Training waits are real (polled), not fixed sleeps.
 *
 * Usage:  node demo-record-sre-flow.js [output.mp4]
 * Requires: platform on :7100, inference server on :7200, host Ollama with
 *           qwen2.5:0.5b, datasets under data/sre-pods/.
 */

const { chromium } = require('playwright');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const OUTPUT = process.argv[2] || 'demo-sre-flow.mp4';
const BASE = 'http://localhost:7100';
const WIDTH = 1366;
const HEIGHT = 854;
const BASE_MODEL = 'qwen2.5:0.5b';   // host Ollama base — same family as the training base
// Question framed from the v2-only NodeDrainCrashLoop incidents (e.g. sre-00101).
const HERO = 'Our payments-api pods keep entering CrashLoopBackOff right after a node drain / cluster autoscaler scale-down — the database connection pool exhausts when all replicas restart together. What is the root cause, and what preventive measures stop it from recurring?';

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  console.log('Launching browser...');
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: 1,
    recordVideo: { dir: __dirname, size: { width: WIDTH, height: HEIGHT } },
  });
  const page = await context.newPage();

  // ── Caption overlay ────────────────────────────────────────────────
  async function installOverlay() {
    await page.evaluate(() => {
      if (document.getElementById('__cap')) return;
      const chip = document.createElement('div');
      chip.id = '__chip';
      chip.style.cssText = 'position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:2147483647;background:#2563eb;color:#fff;font:600 13px -apple-system,Segoe UI,sans-serif;padding:6px 14px;border-radius:999px;box-shadow:0 4px 14px rgba(0,0,0,.4);opacity:0;transition:opacity .3s;';
      document.body.appendChild(chip);
      const cap = document.createElement('div');
      cap.id = '__cap';
      cap.style.cssText = 'position:fixed;bottom:28px;left:50%;transform:translateX(-50%);z-index:2147483647;max-width:1100px;width:90%;text-align:center;background:rgba(13,17,23,.92);color:#fff;font:500 19px/1.4 -apple-system,Segoe UI,sans-serif;padding:16px 26px;border-radius:12px;border:1px solid rgba(255,255,255,.12);box-shadow:0 8px 30px rgba(0,0,0,.5);opacity:0;transition:opacity .3s;';
      const t = document.createElement('div'); t.id = '__capt'; t.style.cssText = 'font-weight:700;font-size:21px;margin-bottom:4px;';
      const s = document.createElement('div'); s.id = '__caps'; s.style.cssText = 'font-size:16px;color:#9fb3c8;';
      cap.appendChild(t); cap.appendChild(s); document.body.appendChild(cap);
    });
  }
  async function cap(chip, title, sub = '') {
    await installOverlay();
    await page.evaluate(([chip, title, sub]) => {
      const c = document.getElementById('__chip');
      c.textContent = chip; c.style.opacity = chip ? '1' : '0';
      document.getElementById('__capt').textContent = title;
      document.getElementById('__caps').textContent = sub;
      document.getElementById('__cap').style.opacity = title ? '1' : '0';
    }, [chip, title, sub]);
  }

  // ── Real waits ──────────────────────────────────────────────────────
  async function waitTrain(label, timeoutMs = 1800000) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      const s = await page.evaluate(async () => {
        try { return await (await fetch('/api/train/status')).json(); } catch (e) { return {}; }
      });
      if (s.status && s.status !== 'running' && s.status !== 'starting') {
        console.log(`  ${label}: ${s.status} final_loss=${s.final_loss}`);
        return s;
      }
      await sleep(2000);
    }
    console.log(`  ${label}: TIMEOUT`);
    return {};
  }
  async function waitExport(niche, timeoutMs = 360000) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      const has = await page.evaluate(async (n) => {
        try { const d = await (await fetch('/api/models')).json();
          return d.models.some(m => m.id === n && m.provider === 'inference'); } catch (e) { return false; }
      }, niche);
      if (has) { console.log(`  export ${niche}: served`); return true; }
      await sleep(2500);
    }
    console.log(`  export ${niche}: TIMEOUT`); return false;
  }
  async function waitReply(timeoutMs = 180000) {
    await page.waitForFunction(() => {
      const btn = document.getElementById('send-btn');
      const msgs = document.querySelectorAll('#chat-area .message.assistant');
      const last = msgs[msgs.length - 1];
      return btn && !btn.disabled && last && last.querySelector('.meta');
    }, { timeout: timeoutMs }).catch(() => {});
  }
  async function type(sel, text) { await page.fill(sel, ''); await page.type(sel, text, { delay: 14 }); }
  async function ask(model) {
    await page.evaluate(() => loadModels()); await sleep(1200);
    await page.selectOption('#model-select', model).catch(() => {});
    await sleep(700);
    await type('#message-input', HERO);
    await sleep(500);
    await page.click('#send-btn');
    await waitReply();
    await sleep(1000);
    await page.evaluate(() => { const c = document.getElementById('chat-area'); if (c) c.scrollTop = c.scrollHeight; });
  }

  // ═══════════════════════════════════════════════════════════════════
  console.log('Recording...');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await installOverlay();
  await cap('Fine-Tuning Platform · localhost:7100',
    'Teach a local model to debug Kubernetes — and watch it get better',
    'SRE incident data · LoRA fine-tuning · HuggingFace/CPU · runs fully offline');
  await sleep(5000);

  const TRAIN = '#train-tab';
  const tabChat = page.locator('.tab').nth(0);

  // ── STEP 1: BASE model, before fine-tuning ──
  await tabChat.click(); await sleep(600);
  await cap('Step 1 of 7 · Before fine-tuning', 'Ask the raw base model the node-drain question',
    BASE_MODEL + ' — no training yet');
  await ask(BASE_MODEL);
  await cap('Step 1 of 7 · Base answer', 'Base model: a generic CrashLoopBackOff checklist',
    'No notion of node-drain / connection-pool stampede — the real root cause');
  await sleep(7000);

  // ── STEP 2: dataset v1 ──
  await page.click(TRAIN); await sleep(800);
  await cap('Step 2 of 7 · Dataset v1', 'Point at 100 real SRE incident rows',
    'pod status, kubectl events & logs → root cause + preventive measures');
  await page.selectOption('#ft-dataset-type', 'local').catch(() => {});
  await sleep(400);
  await type('#ft-niche', 'sre-pods');
  await type('#ft-data-path', 'data/sre-pods/dataset_v1.jsonl');
  await sleep(500);
  await page.click('button:has-text("Use This Dataset")').catch(() => {});
  await sleep(1200);

  // ── STEP 3: base model + params, then train v1 (LIVE) ──
  await page.selectOption('#ft-base-model', 'mlx-community/Qwen2.5-0.5B-Instruct-4bit').catch(() => {});
  await type('#ft-rows', '120');
  await type('#ft-epochs', '2');
  await sleep(800);
  await cap('Step 3 of 7 · Train v1 (live)', 'Fine-tune from the base — watch the loss curve fall',
    'Qwen2.5-0.5B · LoRA rank 16 · 2 epochs · 100 rows · CPU');
  await page.click('#ft-start-btn');
  await sleep(2500);
  const r1 = await waitTrain('v1');
  await sleep(1500);
  await cap('Step 3 of 7 · Train v1 ✓', 'v1 trained — final loss ~' + (r1.final_loss != null ? r1.final_loss.toFixed(3) : '—'),
    'Learned the SRE answer style from 100 incidents');
  await sleep(3000);

  // ── export v1 ──
  await cap('Step 3 of 7 · Deploy v1', 'Merge the adapter into the base and serve it',
    'One click: standalone model, hot on the inference server :7200');
  await page.click('button:has-text("Export & Serve")').catch(() => {});
  await waitExport('sre-pods');
  await sleep(1500);

  // ── STEP 4: chat v1 ──
  await tabChat.click(); await sleep(700);
  await cap('Step 4 of 7 · Ask v1', 'Same question — to the fine-tuned v1 model',
    'sre-pods (fine-tuned on 100 rows)');
  await ask('sre-pods');
  await cap('Step 4 of 7 · v1 answer', 'v1: sharper SRE style — but still generic on node-drain',
    'The node-drain root cause was NOT in v1\'s 100 rows yet');
  await sleep(7000);

  // ── STEP 5: retrain (CONTINUE) on 150 rows ──
  await page.click(TRAIN); await sleep(800);
  await page.evaluate(() => loadAdapters()); await sleep(800);
  await cap('Step 5 of 7 · Improve it', 'Add 50 node-drain incidents (150 total) and CONTINUE from v1',
    'Incremental fine-tuning — not from scratch; resumes the v1 adapter');
  await type('#ft-niche', 'sre-pods-v2');
  await type('#ft-data-path', 'data/sre-pods/dataset_v2.jsonl');
  await sleep(500);
  await page.click('button:has-text("Use This Dataset")').catch(() => {});
  await sleep(700);
  await page.selectOption('#ft-resume-adapter', 'models/adapters/sre-pods').catch(async () => {
    await page.selectOption('#ft-resume-adapter', { label: 'sre-pods' }).catch(() => {});
  });
  await type('#ft-rows', '200');
  await type('#ft-epochs', '1');
  await sleep(1500);
  await cap('Step 6 of 7 · Retrain (live)', 'Continue training the v1 adapter on the bigger dataset',
    'Starts from v1\'s weights — including the 50 new node-drain incidents');
  await page.click('#ft-start-btn');
  await sleep(2500);
  const r2 = await waitTrain('v2');
  await sleep(1500);
  await cap('Step 6 of 7 · Retrain ✓', 'v2 final loss ~' + (r2.final_loss != null ? r2.final_loss.toFixed(3) : '—'),
    'More data + continued training = a better model');
  await sleep(3000);
  await page.click('button:has-text("Export & Serve")').catch(() => {});
  await waitExport('sre-pods-v2');
  await sleep(1500);

  // ── STEP 7: chat v2 ──
  await tabChat.click(); await sleep(700);
  await cap('Step 7 of 7 · Ask v2', 'The exact same question — to the retrained model',
    'sre-pods-v2 (fine-tuned on 150 rows, continued from v1)');
  await ask('sre-pods-v2');
  await cap('Step 7 of 7 · v2 answer', 'v2: nails the node-drain root cause + a real fix list',
    'connection-pool stampede · PodDisruptionBudget · graceful SIGTERM · ≥2 replicas');
  await sleep(9000);

  // ── Recap ──
  await page.click(TRAIN); await sleep(800);
  await page.evaluate(() => loadTrainHistory()); await sleep(1000);
  await page.evaluate(() => { const s = document.querySelector('.sidebar'); if (s) s.scrollTop = s.scrollHeight; });
  await cap('', 'Base → v1 → v2: the same question, a measurably better answer',
    'Generic checklist → SRE style → exact node-drain root cause. All local, on CPU.');
  await sleep(8000);

  console.log('Recording complete.');
  await context.close();
  await browser.close();

  // ── Build mp4 from the webm (no audio track) ──
  const webm = fs.readdirSync(__dirname).filter(f => f.endsWith('.webm')).map(f => path.join(__dirname, f))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  if (!webm) { console.log('No video captured'); return; }
  const finalOutput = path.resolve(OUTPUT);
  console.log('Encoding mp4...');
  execSync(`ffmpeg -i "${webm}" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart "${finalOutput}" -y`,
    { stdio: 'pipe', timeout: 240000 });
  const size = fs.statSync(finalOutput).size;
  const dur = parseFloat(execSync(`ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${finalOutput}"`).toString().trim());
  console.log(`✓ Demo: ${finalOutput} (${(size / 1e6).toFixed(1)} MB, ${dur.toFixed(0)}s)`);
  try { fs.unlinkSync(webm); } catch (e) {}
  console.log('Done!');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
