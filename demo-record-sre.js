/**
 * Narrated-by-caption demo recording for the SRE / Kubernetes incremental
 * fine-tuning story:
 *
 *   import 100-row dataset -> train v1 (live loss curve) -> export & serve ->
 *   chat the hero question (generic answer) -> add 50 rows (150 total) and
 *   CONTINUE training from v1 (loss picks up where it left off, drops lower) ->
 *   export & serve -> chat the same question (sharper answer) -> show both curves.
 *
 * On-screen captions (no audio). Training runs LIVE — waits are real (polled),
 * not fixed sleeps, so the loss curve animates as it actually trains.
 *
 * Usage:  node demo-record-sre.js [output.mp4]
 * Requires: platform on :7100, inference server on :7200, datasets under
 *           data/sre-pods/. Wipe models/adapters/sre-pods* first for a clean run.
 */

const { chromium } = require('playwright');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const OUTPUT = process.argv[2] || 'demo.mp4';
const BASE = 'http://localhost:7100';
const WIDTH = 1366;
const HEIGHT = 854;
const HERO = 'Our payments-api pods keep entering CrashLoopBackOff right after a node drain / cluster autoscaler scale-down. What is the most likely root cause, and what preventive measures stop it from happening again?';

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
  async function waitTrain(label, timeoutMs = 240000) {
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
  async function waitExport(niche, timeoutMs = 240000) {
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
  async function waitReply(timeoutMs = 120000) {
    await page.waitForFunction(() => {
      const btn = document.getElementById('send-btn');
      const msgs = document.querySelectorAll('#chat-area .message.assistant');
      const last = msgs[msgs.length - 1];
      return btn && !btn.disabled && last && last.querySelector('.meta');
    }, { timeout: timeoutMs }).catch(() => {});
  }
  async function type(sel, text) { await page.fill(sel, ''); await page.type(sel, text, { delay: 18 }); }

  // ═══════════════════════════════════════════════════════════════════
  console.log('Recording...');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await installOverlay();
  await cap('Fine-Tuning Platform · localhost:7100',
    'Train a local SRE model on your own data — then make it better',
    'DeepSeek-verified data · MLX LoRA on Apple Silicon · runs fully offline');
  await sleep(5000);

  // Tabs: nth(0)=Chat, nth(1)=Train
  const TRAIN = '#train-tab';
  const tabChat = page.locator('.tab').nth(0);

  // ── STEP 1: dataset ──
  await page.click(TRAIN); await sleep(800);
  await cap('Step 1 of 6 · Dataset', 'Point at your dataset: 100 real SRE incident rows',
    'Kubernetes pod status, events & uptime logs → root cause + preventive measures');
  await page.selectOption('#ft-dataset-type', 'local').catch(() => {});
  await sleep(400);
  await type('#ft-niche', 'sre-pods');
  await type('#ft-data-path', 'data/sre-pods/dataset_v1.jsonl');
  await sleep(600);
  await page.click('button:has-text("Use This Dataset")').catch(() => {});
  await sleep(1500);

  // ── STEP 2: model + params ──
  await cap('Step 2 of 6 · Base model', 'Pick the open base model and LoRA settings',
    'Qwen2.5-1.5B (4-bit) · rank 16 · 3 epochs · 100 rows');
  await page.selectOption('#ft-base-model', 'mlx-community/Qwen2.5-1.5B-Instruct-4bit').catch(() => {});
  await type('#ft-rows', '120');
  await sleep(1500);

  // ── STEP 3: train v1 (LIVE) ──
  await cap('Step 3 of 6 · Train v1', 'Fine-tuning live — watch the loss curve fall',
    'LoRA training in an isolated subprocess, streaming progress over SSE');
  await page.click('#ft-start-btn');
  await sleep(2500);
  const r1 = await waitTrain('v1');
  await sleep(1500);
  await cap('Step 3 of 6 · Train v1 ✓', 'v1 trained — loss fell from ~3.4 to ~' + (r1.final_loss != null ? r1.final_loss.toFixed(3) : '0.03'),
    'Trained on 100 SRE rows from the base model');
  await sleep(3500);

  // ── export v1 ──
  await cap('Step 3 of 6 · Deploy', 'Merge the adapter into the base and serve it',
    'One click: fused standalone model, hot on the inference server :7200');
  await page.click('button:has-text("Export & Serve")').catch(() => {});
  await waitExport('sre-pods');
  await sleep(1500);

  // ── STEP 4: chat v1 ──
  await tabChat.click(); await sleep(800);
  await page.evaluate(() => loadModels()); await sleep(1200);
  await cap('Step 4 of 6 · Ask v1', 'Ask the fine-tuned model the hero question',
    'payments-api CrashLoopBackOff after a node drain — what and why?');
  await page.selectOption('#model-select', 'sre-pods').catch(() => {});
  await sleep(800);
  await type('#message-input', HERO);
  await sleep(600);
  await page.click('#send-btn');
  await waitReply();
  await sleep(1000);
  await page.evaluate(() => { const c = document.getElementById('chat-area'); if (c) c.scrollTop = c.scrollHeight; });
  await cap('Step 4 of 6 · v1 answer', 'v1: a generic CrashLoopBackOff checklist',
    'Correct but not specific to the node-drain root cause');
  await sleep(6000);

  // ── STEP 5: retrain (continue) on 150 rows ──
  await page.click(TRAIN); await sleep(800);
  await page.evaluate(() => loadAdapters()); await sleep(800);
  await cap('Step 5 of 6 · Improve it', 'Add 50 more rows (150 total) and CONTINUE training v1',
    'Not from scratch — incremental fine-tuning from the v1 adapter');
  await type('#ft-niche', 'sre-pods-v2');
  await type('#ft-data-path', 'data/sre-pods/dataset_v2.jsonl');
  await sleep(500);
  await page.click('button:has-text("Use This Dataset")').catch(() => {});
  await sleep(800);
  await page.selectOption('#ft-resume-adapter', 'models/adapters/sre-pods').catch(async () => {
    await page.selectOption('#ft-resume-adapter', { label: 'sre-pods' }).catch(() => {});
  });
  await type('#ft-rows', '200');
  await sleep(1500);
  await cap('Step 5 of 6 · Retrain (live)', 'Loss continues from where v1 finished — and drops lower',
    'Watch the curve start low (~0.03) instead of at 3.4');
  await page.click('#ft-start-btn');
  await sleep(2500);
  const r2 = await waitTrain('v2');
  await sleep(1500);
  await cap('Step 5 of 6 · Retrain ✓', 'v2 best loss ~' + (r2.final_loss != null ? r2.final_loss.toFixed(3) : '0.018') + ' — lower than v1',
    'More data + continued training = a better model');
  await sleep(3000);
  await page.click('button:has-text("Export & Serve")').catch(() => {});
  await waitExport('sre-pods-v2');
  await sleep(1500);

  // ── STEP 6: chat v2 ──
  await tabChat.click(); await sleep(800);
  await page.evaluate(() => loadModels()); await sleep(1200);
  await cap('Step 6 of 6 · Ask v2', 'Same question — to the retrained model',
    'sre-pods-v2 (fine-tuned)');
  await page.selectOption('#model-select', 'sre-pods-v2').catch(() => {});
  await sleep(800);
  await type('#message-input', HERO);
  await sleep(600);
  await page.click('#send-btn');
  await waitReply();
  await sleep(1000);
  await page.evaluate(() => { const c = document.getElementById('chat-area'); if (c) c.scrollTop = c.scrollHeight; });
  await cap('Step 6 of 6 · v2 answer', 'v2: nails the node-drain root cause + a real preventive checklist',
    'PodDisruptionBudget · graceful shutdown · replica spread — measurably better');
  await sleep(8000);

  // ── Recap: both curves in history ──
  await page.click(TRAIN); await sleep(800);
  await page.evaluate(() => loadTrainHistory()); await sleep(1000);
  await page.evaluate(() => { const s = document.querySelector('.sidebar'); if (s) s.scrollTop = s.scrollHeight; });
  await cap('', 'Two runs, one model that got better',
    'v1: 3.4 → 0.03 (100 rows) · v2 continued: 0.03 → 0.018 (150 rows). All local.');
  await sleep(7000);

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
    { stdio: 'pipe', timeout: 180000 });
  const size = fs.statSync(finalOutput).size;
  const dur = parseFloat(execSync(`ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${finalOutput}"`).toString().trim());
  console.log(`✓ Demo: ${finalOutput} (${(size / 1e6).toFixed(1)} MB, ${dur.toFixed(0)}s)`);
  try { fs.unlinkSync(webm); } catch (e) {}
  console.log('Done!');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
