/**
 * Demo recording — captures the Fine-Tuning Platform browser UI.
 * Uses Playwright video recording + ffmpeg conversion.
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
  //  1. LOAD PLATFORM — show the main UI
  // ═══════════════════════════════════════════════════════════
  console.log('1. Loading platform...');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await sleep(3000);

  // ═══════════════════════════════════════════════════════════
  //  2. SHOW INFERENCE SERVER — expand the sidebar section
  // ═══════════════════════════════════════════════════════════
  console.log('2. Inference server...');
  const docsHeaders = await page.$$('.docs-header');
  if (docsHeaders.length > 0) {
    // First docs-header is "Inference Server"
    await docsHeaders[0].click();
    await sleep(1500);
  }

  // ═══════════════════════════════════════════════════════════
  //  3. SHOW LEADERBOARD — scroll sidebar
  // ═══════════════════════════════════════════════════════════
  console.log('3. Leaderboard...');
  await sleep(1000);

  // ═══════════════════════════════════════════════════════════
  //  4. OPEN DOCUMENTATION — browse sections
  // ═══════════════════════════════════════════════════════════
  console.log('4. Documentation...');
  if (docsHeaders.length > 1) {
    // Second docs-header is "Documentation"
    await docsHeaders[1].click();
    await sleep(1000);

    // Click first tree item: "🚀 The Pitch"
    const firstItem = await page.$('.docs-tree-item .label');
    if (firstItem) {
      await firstItem.click();
      await sleep(3000);
    }

    // Close docs panel
    const closeBtn = await page.$('.panel-close');
    if (closeBtn) await closeBtn.click();
    await sleep(1000);
  }

  // ═══════════════════════════════════════════════════════════
  //  5. SWITCH TO TRAIN TAB — show the step wizard
  // ═══════════════════════════════════════════════════════════
  console.log('5. Train tab...');
  const tabs = await page.$$('.tab');
  if (tabs.length > 1) {
    await tabs[1].click(); // "Train" tab
    await sleep(2000);

    // Scroll through form fields to show step indicator
    const nicheInput = await page.$('#ft-niche');
    if (nicheInput) {
      await nicheInput.click();
      await sleep(500);
      await nicheInput.fill('insurance-k3s');
      await sleep(500);
    }

    // Click on model selector to advance step
    const baseModel = await page.$('#ft-base-model');
    if (baseModel) {
      await baseModel.click();
      await sleep(1000);
    }

    // Click hyperparameters
    const loraRank = await page.$('#ft-lora-rank');
    if (loraRank) {
      await loraRank.click();
      await sleep(1000);
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  6. SWITCH BACK TO CHAT — type a question
  // ═══════════════════════════════════════════════════════════
  console.log('6. Chat...');
  await sleep(1000);
  const chatTab = await page.$('.tab'); // First tab is Chat
  if (chatTab) await chatTab.click();
  await sleep(1000);

  // ═══════════════════════════════════════════════════════════
  //  7. TEST INFERENCE — via the inference server API
  // ═══════════════════════════════════════════════════════════
  console.log('7. Testing inference...');
  await sleep(500);

  // ═══════════════════════════════════════════════════════════
  //  8. CLOSE AND CONVERT
  // ═══════════════════════════════════════════════════════════
  console.log('Closing...');
  await sleep(1000);
  await browser.close();

  // Get video path
  const videoDir = fs.readdirSync(__dirname).filter(f => f.endsWith('.webm'));
  const videoPath = videoDir.length > 0
    ? path.join(__dirname, videoDir[0])
    : null;

  if (videoPath) {
    const finalOutput = path.resolve(OUTPUT);
    console.log(`Converting ${videoPath} → ${finalOutput}...`);
    try {
      execSync(
        `ffmpeg -i "${videoPath}" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p "${finalOutput}" -y 2>/dev/null`,
        { stdio: 'pipe', timeout: 30000 }
      );
      const size = fs.statSync(finalOutput).size;
      console.log(`✓ Demo video: ${finalOutput} (${(size/1e6).toFixed(1)} MB)`);
      fs.unlinkSync(videoPath);
    } catch (err) {
      console.error(`ffmpeg failed: ${err.message}`);
      console.log(`Raw video: ${videoPath}`);
    }
  } else {
    console.log('No .webm video found');
  }

  console.log('Done!');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
