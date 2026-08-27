#!/usr/bin/env node

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const args = parseArgs(process.argv.slice(2));
const port = String(args.port || process.env.ACTIVE_WEB_PORT || '3000');
const apiUrl = String(args.apiUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
const lineageRoot = path.resolve(String(args.lineageRoot || process.env.EVOLUTION_LINEAGE_ROOT || path.join(root, 'evolution')));
const pollMs = Number(args.pollMs || process.env.ACTIVE_WEB_POLL_MS || 1500);
const nextBin = path.join(root, 'node_modules', 'next', 'dist', 'bin', 'next');
const shouldStartApi = args.startApi !== 'false' && process.env.ACTIVE_WEB_START_API !== 'false';

let child = null;
let apiChild = null;
let runningKey = '';
let stopping = false;

if (!fs.existsSync(nextBin)) {
  console.error(`[active-web] Next.js binary not found: ${nextBin}`);
  console.error('[active-web] Run npm install in the root project first.');
  process.exit(1);
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith('--')) {
      continue;
    }
    const [rawKey, inlineValue] = value.slice(2).split('=', 2);
    const key = rawKey.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    parsed[key] = inlineValue ?? values[index + 1];
    if (inlineValue === undefined) {
      index += 1;
    }
  }
  return parsed;
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function activeRepo() {
  const activePath = path.join(lineageRoot, 'active.json');
  const active = readJson(activePath);
  const candidate = active?.child_repo ? path.resolve(String(active.child_repo)) : root;
  const appDir = path.join(candidate, 'apps', 'web');
  if (!fs.existsSync(path.join(appDir, 'package.json'))) {
    return {
      repo: root,
      appDir: path.join(root, 'apps', 'web'),
      label: 'root fallback',
      generation: null,
    };
  }
  return {
    repo: candidate,
    appDir,
    label: active?.active_generation ? `agent-${String(active.active_generation).padStart(3, '0')}` : 'root',
    generation: active?.active_generation ?? null,
  };
}

function startNext(target) {
  const env = {
    ...process.env,
    NEXT_PUBLIC_API_URL: apiUrl,
  };
  const spawned = spawn(process.execPath, [nextBin, 'dev', '-p', port], {
    cwd: target.appDir,
    env,
    detached: true,
    stdio: 'inherit',
  });
  child = spawned;
  runningKey = target.repo;
  console.log(`[active-web] serving ${target.label} from ${target.appDir}`);
  console.log(`[active-web] url http://localhost:${port}`);
  spawned.on('exit', (code, signal) => {
    if (child === spawned) {
      child = null;
    }
    if (!stopping && signal !== 'SIGTERM') {
      console.log(`[active-web] Next.js exited with code=${code ?? 'null'} signal=${signal ?? 'null'}`);
    }
  });
}

function apiHealthUrl() {
  try {
    return new URL('/api/settings', apiUrl).toString();
  } catch {
    return '';
  }
}

function isLocalApiUrl() {
  try {
    const parsed = new URL(apiUrl);
    return ['localhost', '127.0.0.1', '::1'].includes(parsed.hostname);
  } catch {
    return false;
  }
}

async function apiReachable() {
  const healthUrl = apiHealthUrl();
  if (!healthUrl) {
    return false;
  }
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1000);
    const response = await fetch(healthUrl, { cache: 'no-store', signal: controller.signal });
    clearTimeout(timeout);
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureApi() {
  if (!shouldStartApi || !isLocalApiUrl()) {
    return;
  }
  if (await apiReachable()) {
    return;
  }

  const apiDir = path.join(root, 'apps', 'api');
  const uvicorn = path.join(apiDir, '.venv', 'bin', 'uvicorn');
  if (!fs.existsSync(uvicorn)) {
    console.log(`[active-web] API is not reachable at ${apiUrl}; uvicorn not found at ${uvicorn}`);
    return;
  }

  const parsed = new URL(apiUrl);
  const apiPort = parsed.port || (parsed.protocol === 'https:' ? '443' : '80');
  apiChild = spawn(uvicorn, ['app.main:app', '--reload', '--host', '127.0.0.1', '--port', apiPort], {
    cwd: apiDir,
    env: process.env,
    detached: true,
    stdio: 'inherit',
  });
  console.log(`[active-web] started API at ${apiUrl}`);
  apiChild.on('exit', (code, signal) => {
    if (apiChild) {
      apiChild = null;
    }
    if (!stopping && signal !== 'SIGTERM') {
      console.log(`[active-web] API exited with code=${code ?? 'null'} signal=${signal ?? 'null'}`);
    }
  });
}

function stopNext() {
  if (!child) {
    return Promise.resolve();
  }
  const current = child;
  child = null;
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      if (!current.killed) {
        current.kill('SIGKILL');
      }
      resolve();
    }, 5000);
    current.once('exit', () => {
      clearTimeout(timeout);
      resolve();
    });
    terminateProcess(current, 'SIGTERM');
  });
}

function terminateProcess(target, signal) {
  if (!target?.pid) {
    return;
  }
  try {
    process.kill(-target.pid, signal);
  } catch {
    try {
      target.kill(signal);
    } catch {
      // Process is already gone.
    }
  }
}

async function sync() {
  if (stopping) {
    return;
  }
  const target = activeRepo();
  if (target.repo === runningKey && child) {
    return;
  }
  if (child) {
    console.log(`[active-web] active generation changed; restarting from ${target.label}`);
    await stopNext();
  }
  startNext(target);
}

async function shutdown(signal) {
  stopping = true;
  console.log(`[active-web] received ${signal}; stopping`);
  await stopNext();
  if (apiChild) {
    terminateProcess(apiChild, 'SIGTERM');
  }
  process.exit(0);
}

process.on('SIGINT', () => void shutdown('SIGINT'));
process.on('SIGTERM', () => void shutdown('SIGTERM'));

await ensureApi();
await sync();
setInterval(() => void sync(), Math.max(500, pollMs));
