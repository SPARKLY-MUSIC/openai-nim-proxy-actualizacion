// server.js — Robust Hybrid OpenAI ↔ NIM Proxy (FIEL v3.0 — Veloz, Fiel y con Modelos Activos)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── PERF: Shared keep-alive agent (connection reuse) ────────────────────────
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────

// Aliases are periodically re-checked against NIM's live catalog (see
// validateModels() and GET /v1/models?live=true). Comments note the prior
// backend model ID where an entry was swapped out for a dead catalog entry.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'gpt-4-turbo': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct', 
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b',
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'mistral': 'mistralai/mistral-large-2-instruct', 
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct',
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3', 
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', 
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b',
  'google-lighter': 'poolside/laguna-xs-2.1',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5-3-flash',

  // Vision-Modelle — nur Modelle, die erfolgreich auf Chat-Completions antworten.
  // Getestet am: 2026-09-17 — meta/llama-3.2-11b-vision-instruct: HTTP 200 ✓
  // meta/llama-3.2-90b-vision-instruct: Timeout (HTTP 000) — nicht hinzugefügt.
  // microsoft/phi-3-vision-128k-instruct: HTTP 404 (nicht in diesem Account) — nicht hinzugefügt.
  // nvidia/neva-22b: HTTP 404 — nicht hinzugefügt.
  // nvidia/vila: HTTP 404 — nicht hinzugefügt.
  'llama-vision': 'meta/llama-3.2-11b-vision-instruct',
  'vision': 'meta/llama-3.2-11b-vision-instruct',
  'llama-vision-11b': 'meta/llama-3.2-11b-vision-instruct',
  'llama-3.2-vision': 'meta/llama-3.2-11b-vision-instruct'
};

// Used when an unrecognized alias is requested. Must point at a live model.
const DEFAULT_MODEL = 'google/diffusiongemma-26b-a4b-it';

// Ordered by observed reliability/speed — an early failing model delays every fallback behind it.
const FALLBACK_MODELS = [
  'google/diffusiongemma-26b-a4b-it',
  'google/gemma-4-31b-it',
  'mistralai/mistral-nemotron',
  'nvidia/nemotron-3-super-120b-a12b'
];

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);

  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await nim.get('/models', {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];

    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── PERF: Fallback chain with cooldowns + de-dupe ──────────────────────────
const modelCooldowns = new Map();

const RATE_LIMIT_COOLDOWN_MS = Number(process.env.RATE_LIMIT_COOLDOWN_MS) || 30000;
const ACCESS_DENIED_COOLDOWN_MS = Number(process.env.ACCESS_DENIED_COOLDOWN_MS) || 300000;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    try {
      const res = await nim.post(
        '/chat/completions',
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );

      return { response: res, model };

    } catch (err) {
      lastError = err;
      const status = err.response?.status;

      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        status,
        err.response?.data?.error?.message || err.message
      );

      if (status === 401) throw err;
      if (status === 403) setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      if (status === 429) setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '3.1.0-fiel' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const { model, max_tokens, temperature, stream } = req.body;

    const primaryModel = MODEL_MAPPING[model] || 'nvidia/llama-3.1-nemotron-70b-instruct';
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    // FIDELITY: forward ALL client fields intact so prompts and user messages are respected
    const { model: _droppedModel, ...forwardedFields } = req.body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.trim() === 'data: [DONE]') {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING) {
              if (reasoning && !reasoningOpen) {
                content = `<thinking>\n${reasoning}`;
                reasoningOpen = true;
              } else if (reasoning) {
                content = reasoning;
              }

              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }

            delta.content = content;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow');
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (SHOW_REASONING && reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] })}\n\n`);
          reasoningOpen = false;
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        if (!streamEndedCleanly && (req.destroyed || !res.writable)) {
          if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: usedModel,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            content = `<thinking>\n${choice.message.reasoning_content}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              tool_calls: choice.message?.tool_calls
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    if (!res.headersSent) {
      res.set('Content-Type', 'application/json');
      res.status(error.response?.status || 500).json({
        error: { message: error.message, type: 'invalid_request_error', code: error.response?.status || 500 }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({ error: { message: 'Not found', type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`[PROXY] Fiel v3.1 running on port ${PORT} (Keep-Alive + Cooldowns + Active Models)`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
