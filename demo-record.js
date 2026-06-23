/**
 * Narrated demo recording — captures the Fine-Tuning Platform UI
 * with voiceover narration synced to screen actions.
 *
 * Usage:  node demo-record.js [output.mp4]
 * Requires: platform on :7100, inference server on :7200
 */

const { chromium } = require('playwright');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const OUTPUT = process.argv[2] || 'demo.mp4';
const BASE = 'http://localhost:7100';
const WIDTH = 1280;
const HEIGHT = 800;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  console.log('Launching browser...');
  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: 1,
    recordVideo: { dir: __dirname, size: { width: WIDTH, height: HEIGHT } },
  });

  const page = await context.newPage();

  // ═══════════════════════════════════════════════════════════
  //  RECORD THE SCREEN INTERACTION
  // ═══════════════════════════════════════════════════════════

  // 0-4s: Load platform
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await sleep(4000);

  // 4-10s: Show inference server panel + docs
  const docsHeaders = await page.$$('.docs-header');
  if (docsHeaders.length > 0) {
    await docsHeaders[0].click(); // Inference
    await sleep(2000);
  }
  if (docsHeaders.length > 1) {
    await docsHeaders[1].click(); // Docs
    await sleep(1000);
    const firstItem = await page.$('.docs-tree-item .label');
    if (firstItem) { await firstItem.click(); await sleep(3000); }
  }

  // 10-16s: Switch to Train tab, show niche
  const closeBtn = await page.$('.panel-close');
  if (closeBtn) await closeBtn.click();
  await sleep(1000);

  const tabs = await page.$$('.tab');
  if (tabs.length > 1) await tabs[1].click();
  await sleep(1000);

  const nicheInput = await page.$('#ft-niche');
  if (nicheInput) {
    await nicheInput.click();
    await sleep(500);
    for (const c of 'insurance-k3s') await page.keyboard.type(c, { delay: 20 });
  }
  await sleep(3500);

  // 16-25s: Generate dataset
  const datasetType = await page.$('#ft-dataset-type');
  if (datasetType) { await datasetType.selectOption('bigset'); await sleep(1500); }

  const descInput = await page.$('#ft-desc');
  if (descInput) {
    await descInput.click();
    for (const c of 'K3s cluster pods and services for insurance database') {
      await page.keyboard.type(c, { delay: 8 });
    }
    await sleep(2000);
  }

  const genBtn = await page.$('#ft-gen-btn');
  if (genBtn) { await genBtn.click(); await sleep(5000); }

  // 25-33s: Training
  if (datasetType) { await datasetType.selectOption('local'); await sleep(1500); }

  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await btn.textContent();
    if (text.includes('Use This Dataset')) { await btn.click(); break; }
  }
  await sleep(1500);

  const startBtn = await page.$('#ft-start-btn');
  if (startBtn) {
    const disabled = await startBtn.getAttribute('disabled');
    if (disabled === null) { await startBtn.click(); }
  }
  await sleep(8000);

  // 33-40s: Chat with v1
  if (tabs.length > 0) await tabs[0].click();
  await sleep(1000);

  const msgInput = await page.$('#message-input');
  if (msgInput) {
    await msgInput.click();
    for (const c of 'How do I mount a PersistentVolumeClaim in K3s for insurance data?') {
      await page.keyboard.type(c, { delay: 5 });
    }
    await sleep(1000);
    const sendBtn = await page.$('#send-btn');
    if (sendBtn) await sendBtn.click();
    await sleep(5000);
  }

  // 40-49s: Train v2 (retrain)
  if (tabs.length > 1) await tabs[1].click();
  await sleep(2000);

  // Start training v2
  if (startBtn) {
    const disabled = await startBtn.getAttribute('disabled');
    if (disabled === null) { await startBtn.click(); }
  }
  await sleep(9000);

  // 49-56s: Chat with v2
  if (tabs.length > 0) await tabs[0].click();
  await sleep(1000);

  if (msgInput) {
    await msgInput.click();
    for (const c of 'How do I mount a PersistentVolumeClaim in K3s for insurance data?') {
      await page.keyboard.type(c, { delay: 5 });
    }
    await sleep(1000);
    const sendBtn = await page.$('#send-btn');
    if (sendBtn) await sendBtn.click();
    await sleep(5000);
  }

  // 56-68s: Show leaderboard
  await page.evaluate(() => {
    const s = document.querySelector('.sidebar');
    if (s) s.scrollTop = s.scrollHeight;
  });
  await sleep(12000);

  console.log('Recording complete.');
  await browser.close();

  // ═══════════════════════════════════════════════════════════
  //  BUILD FINAL VIDEO WITH AUDIO OVERLAY
  // ═══════════════════════════════════════════════════════════
  const webmFiles = fs.readdirSync(__dirname).filter(f => f.endsWith('.webm'));
  const videoPath = webmFiles.length > 0 ? path.join(__dirname, webmFiles[0]) : null;
  if (!videoPath) { console.log('No video found'); return; }

  const rawMp4 = videoPath.replace('.webm', '_raw.mp4');
  const finalOutput = path.resolve(OUTPUT);
  const audioDir = path.join(__dirname, 'demo-assets');

  // Step 1: Convert webm to mp4
  console.log('Converting to mp4...');
  execSync(`ffmpeg -i "${videoPath}" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p "${rawMp4}" -y`,
    { stdio: 'pipe', timeout: 30000 });

  // Step 2: Concatenate all audio segments into one
  const concatList = path.join(audioDir, '_list.txt');
  const audioFiles = fs.readdirSync(audioDir)
    .filter(f => f.endsWith('.aiff'))
    .sort()
    .map(f => path.join(audioDir, f));

  const listContent = audioFiles.map(f => `file '${f}'`).join('\n');
  fs.writeFileSync(concatList, listContent);

  const combinedAudio = path.join(audioDir, '_combined.aac');
  execSync(
    `ffmpeg -f concat -safe 0 -i "${concatList}" -c:a aac -b:a 128k "${combinedAudio}" -y`,
    { stdio: 'pipe', timeout: 15000 }
  );
  fs.unlinkSync(concatList);

  // Step 3: Overlay audio on video
  console.log('Mixing narration with video...');
  execSync(
    `ffmpeg -i "${rawMp4}" -i "${combinedAudio}" ` +
    `-c:v copy -c:a aac -b:a 128k -map 0:v:0 -map 1:a:0 -shortest "${finalOutput}" -y`,
    { stdio: 'pipe', timeout: 30000 }
  );

  const size = fs.statSync(finalOutput).size;
  const dur = parseFloat(execSync(
    `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${finalOutput}"`
  ).toString().trim());
  const audioDur = parseFloat(execSync(
    `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${combinedAudio}"`
  ).toString().trim());

  console.log(`✓ Demo: ${finalOutput} (${(size/1e6).toFixed(1)} MB, ${dur.toFixed(0)}s video + ${audioDur.toFixed(0)}s audio)`);

  // Cleanup
  try { fs.unlinkSync(videoPath); } catch(e) {}
  try { fs.unlinkSync(rawMp4); } catch(e) {}
  try { fs.unlinkSync(combinedAudio); } catch(e) {}

  console.log('Done!');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
