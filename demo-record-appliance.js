/**
 * Appliance walkthrough recording (no voiceover, on-screen captions). Shows the
 * NEW SRE appliance data on :7100: live preset card → dataset DROPDOWN → 2 training
 * trials (v1, then v2 continued) → the OOM answer improving on the 2nd iteration.
 *
 * Usage: node demo-record-appliance.js [out.mp4]
 * Requires: platform on :7100 with sre_qa_v1/v2 datasets + a base/ollama chat model.
 */
const { chromium } = require('playwright');
const { execSync } = require('child_process');
const path = require('path'); const fs = require('fs');

const OUT = process.argv[2] || 'demo-appliance.mp4';
const BASE = 'http://localhost:7100';
const W = 1366, H = 854;
const V1 = 'data/sre-tables-train/sre_qa_v1.jsonl';
const V2 = 'data/sre-tables-train/sre_qa_v2.jsonl';
const HERO = 'Our payments-api pod in payments was flagged for OOM Risk at high risk. Give the full root cause and remediation.';
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: __dirname, size: { width: W, height: H } } });
  const page = await ctx.newPage();

  async function overlay() {
    await page.evaluate(() => {
      if (document.getElementById('__cap')) return;
      const chip = document.createElement('div'); chip.id = '__chip';
      chip.style.cssText = 'position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:2147483647;background:#2563eb;color:#fff;font:600 13px -apple-system,sans-serif;padding:6px 14px;border-radius:999px;opacity:0;transition:.3s';
      document.body.appendChild(chip);
      const cap = document.createElement('div'); cap.id = '__cap';
      cap.style.cssText = 'position:fixed;bottom:26px;left:50%;transform:translateX(-50%);z-index:2147483647;max-width:1100px;width:90%;text-align:center;background:rgba(13,17,23,.93);color:#fff;font:500 19px/1.4 -apple-system,sans-serif;padding:15px 26px;border-radius:12px;border:1px solid rgba(255,255,255,.12);opacity:0;transition:.3s';
      const t = document.createElement('div'); t.id = '__t'; t.style.cssText = 'font-weight:700;font-size:21px;margin-bottom:3px'; cap.appendChild(t);
      const s = document.createElement('div'); s.id = '__s'; s.style.cssText = 'font-size:15px;color:#9fb3c8'; cap.appendChild(s);
      document.body.appendChild(cap);
    });
  }
  async function cap(chip, t, s = '') {
    await overlay();
    await page.evaluate(([chip, t, s]) => {
      const c = document.getElementById('__chip'); c.textContent = chip; c.style.opacity = chip ? '1' : '0';
      document.getElementById('__t').textContent = t; document.getElementById('__s').textContent = s;
      document.getElementById('__cap').style.opacity = t ? '1' : '0';
    }, [chip, t, s]);
  }
  async function waitTrain(label, to = 240000) {
    const t0 = Date.now();
    while (Date.now() - t0 < to) {
      const s = await page.evaluate(async () => { try { return await (await fetch('/api/train/status')).json(); } catch (e) { return {}; } });
      if (s.status && s.status !== 'running' && s.status !== 'starting') { console.log(`  ${label}: ${s.status} final_loss=${s.final_loss}`); return s; }
      await sleep(2000);
    } console.log(`  ${label}: TIMEOUT`); return {};
  }
  async function waitExport(niche, to = 240000) {
    const t0 = Date.now();
    while (Date.now() - t0 < to) {
      const ok = await page.evaluate(async (n) => { try { return (await (await fetch('/api/models')).json()).models.some(m => m.id === n && m.provider === 'inference'); } catch (e) { return false; } }, niche);
      if (ok) { console.log(`  export ${niche}: served`); return true; } await sleep(2500);
    } return false;
  }
  async function waitReply(to = 120000) {
    await page.waitForFunction(() => { const b = document.getElementById('send-btn'); const m = document.querySelectorAll('#chat-area .message.assistant'); const l = m[m.length - 1]; return b && !b.disabled && l && l.querySelector('.meta'); }, { timeout: to }).catch(() => {});
  }
  const tabChat = () => page.locator('.tab').nth(0).click();
  const tabTrain = () => page.click('#train-tab');
  async function type(sel, t) { await page.fill(sel, ''); await page.type(sel, t, { delay: 14 }); }

  console.log('recording…');
  await page.goto(BASE, { waitUntil: 'networkidle' }); await overlay();
  await cap('Fine-Tuning Appliance · localhost:7100', 'A self-contained SRE appliance — data, models & training in one app', 'no internet · bundled Kubernetes observability data'); await sleep(4500);

  // 1. live data via a preset card
  await cap('Step 1 · Live data', 'Tap a preset card — it queries the bundled tables (no model needed)', 'the 43 SRE tables are loaded into a local DuckDB on boot');
  await page.click('text=Top OOM offenders').catch(() => {}); await sleep(4500);

  // 2. dataset dropdown + train trial 1
  await tabTrain(); await sleep(800);
  await cap('Step 2 · Pick a dataset', 'Choose the SRE training set from the dropdown', 'sre_qa_v1 — 141 rows');
  await type('#ft-niche', 'sre-asst-demo');
  await page.selectOption('#ft-dataset-pick', V1).catch(() => {}); await sleep(1200);
  await page.selectOption('#ft-base-model', 'mlx-community/Qwen2.5-1.5B-Instruct-4bit').catch(() => {});
  await type('#ft-rows', '300'); await sleep(1000);
  await cap('Step 3 · Train (trial 1)', 'Fine-tune on v1 — watch the loss curve', '');
  await page.click('#ft-start-btn'); await sleep(2500);
  const r1 = await waitTrain('v1'); await sleep(1500);
  await cap('Step 3 · Trial 1 done', 'v1 trained (loss ' + (r1.final_loss != null ? r1.final_loss.toFixed(3) : '~') + ') — now serve it', '');
  await page.click('button:has-text("Export & Serve")').catch(() => {}); await waitExport('sre-asst-demo'); await sleep(1200);

  // 3. ask the hero question (answer A)
  await tabChat(); await sleep(700); await page.evaluate(() => loadModels()); await sleep(1200);
  await cap('Step 4 · Ask v1', 'Ask the trial-1 model the OOM question', '');
  await page.selectOption('#model-select', 'sre-asst-demo').catch(() => {}); await sleep(600);
  await type('#message-input', HERO); await sleep(500); await page.click('#send-btn'); await waitReply(); await sleep(800);
  await page.evaluate(() => { const c = document.getElementById('chat-area'); if (c) c.scrollTop = c.scrollHeight; });
  await cap('Step 4 · v1 answer', 'Trial 1: a short, generic remediation', ''); await sleep(6000);

  // 4. retrain trial 2 (continue) on v2
  await tabTrain(); await sleep(700); await page.evaluate(() => loadAdapters()); await sleep(800);
  await cap('Step 5 · Add data + retrain', 'Pick v2 (more OOM examples) and CONTINUE from the v1 model', 'sre_qa_v2 — 241 rows · incremental fine-tuning');
  await type('#ft-niche', 'sre-asst-demo-v2');
  await page.selectOption('#ft-dataset-pick', V2).catch(() => {}); await sleep(1000);
  await page.selectOption('#ft-resume-adapter', 'models/adapters/sre-asst-demo').catch(() => {});
  await type('#ft-rows', '300'); await sleep(1200);
  await cap('Step 5 · Train (trial 2)', 'Loss continues from where v1 left off', '');
  await page.click('#ft-start-btn'); await sleep(2500);
  const r2 = await waitTrain('v2'); await sleep(1500);
  await cap('Step 5 · Trial 2 done', 'v2 best loss ' + (r2.final_loss != null ? r2.final_loss.toFixed(3) : '~') + ' — serve it', '');
  await page.click('button:has-text("Export & Serve")').catch(() => {}); await waitExport('sre-asst-demo-v2'); await sleep(1200);

  // 5. ask again (answer B — improved)
  await tabChat(); await sleep(700); await page.evaluate(() => loadModels()); await sleep(1200);
  await cap('Step 6 · Ask v2', 'Same question, retrained model', '');
  await page.selectOption('#model-select', 'sre-asst-demo-v2').catch(() => {}); await sleep(600);
  await type('#message-input', HERO); await sleep(500); await page.click('#send-btn'); await waitReply(); await sleep(800);
  await page.evaluate(() => { const c = document.getElementById('chat-area'); if (c) c.scrollTop = c.scrollHeight; });
  await cap('Step 6 · v2 answer', 'Trial 2: fuller root cause + remediation + prevention', 'more data → better inference'); await sleep(8000);

  await tabTrain(); await sleep(600); await page.evaluate(() => loadTrainHistory()); await sleep(800);
  await cap('', 'Two trials, one model that got better — all local', 'v1 → v2 continued · runs the same on an EKS cluster'); await sleep(6000);

  console.log('done recording'); await ctx.close(); await browser.close();

  const webm = fs.readdirSync(__dirname).filter(f => f.endsWith('.webm')).map(f => path.join(__dirname, f)).sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  if (!webm) { console.log('no video'); return; }
  const final = path.resolve(OUT);
  execSync(`ffmpeg -i "${webm}" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart "${final}" -y`, { stdio: 'pipe', timeout: 180000 });
  console.log('✓', final, (fs.statSync(final).size / 1e6).toFixed(1) + ' MB');
  try { fs.unlinkSync(webm); } catch (e) {}
}
main().catch(e => { console.error('Fatal:', e); process.exit(1); });
