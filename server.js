// server.js — Ultra-Fast & Robust OpenAI-compatible proxy for NVIDIA NIM
// Fusiona la velocidad, keep-alive, razonamiento y tool-recovery de "Rápido"
// con la seguridad robusta, manejo de buffers y validación de "Fiel".

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');
const { SHOW_REASONING, getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ──────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const DEBUG_MODE = process.env.DEBUG_MODE === 'true';
const MAX_TOKENS_LIMIT = 65536;

// Shared keep-alive agent for maximum speed and connection reuse
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 180000;
const REASONING_REQUEST_TIMEOUT_MS = Number(process.env.REASONING_REQUEST_TIMEOUT_MS) || 480000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');
if (DEBUG_MODE) console.log('[CONFIG] Debug mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.2': 'z-ai/glm-5.2',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

const DEFAULT_MODEL = 'google/gemma-4-31b-it';

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Middleware ─────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '50mb' }));

app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: {
        message: 'Invalid JSON in request body',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }
  next(err);
});

// Secure token extraction (Fiel fix)
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
  if (req.path === '/health' || req.path === '/v1/models' || req.path === '/') {
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

// ─── Validation ─────────────────────────────────────────────────────────

async function fetchLiveModelIds() {
  const response = await nim.get('/models', {
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    timeout: VALIDATION_TIMEOUT_MS
  });
  return new Set((response.data.data || []).map(m => m.id));
}

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');
  try {
    const availableModels = await fetchLiveModelIds();
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
    description: `${invalidModels.length} model(s) failed validation.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, { embeds: [embed], username: 'NIM Proxy Monitor' }, { timeout: 5000 });
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ─────────────────────────────────────────

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

// ─── Helper: Fallback Chain ──────────────────────────────────────────────

const modelCooldowns = new Map();
const RATE_LIMIT_COOLDOWN_MS = 30000;
const ACCESS_DENIED_COOLDOWN_MS = 300000;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? REASONING_REQUEST_TIMEOUT_MS
    : REQUEST_TIMEOUT_MS;

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };

    try {
      const res = await nim.post('/chat/completions', fullRequest, {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: baseRequest.stream ? 'stream' : 'json',
        timeout: timeoutMs
      });
      return { response: res, model };
    } catch (err) {
      lastError = err;
      const status = err.response?.status;

      console.warn(`[FALLBACK] Model failed: ${model}`, status, err.response?.data?.error?.message || err.message);

      if (status === 401) throw err;
      if (status === 403) setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      if (status === 429) setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ───────────────────────────────────────────────────────────────

app.get('/', (req, res) => {
  res.type('html').send(`<!DOCTYPE html><html><body style="background:#0b0f14;color:#e6edf3;font-family:sans-serif;text-align:center;padding:50px;"><h1>Fiel-Fast Proxy is up.</h1></body></html>`);
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.7.0-hybrid' });
});

app.get('/v1/models', async (req, res) => {
  res.json({
    object: 'list',
    data: Object.keysMODEL_MAPPING = Object.keys(MODEL_MAPPING).map(id => ({
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
    const { model, max_tokens, temperature, stream, reasoning_effort } = req.body;

    let primaryModel = MODEL_MAPPING[model] || DEFAULT_MODEL;
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    const { model: _dm, reasoning_effort: _dre, ...forwardedFields } = req.body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      reasoning_effort,
      !!req.body.tools
    );

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);
      const toolRecovery = new ToolCallStreamRecovery();

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) upstreamStream.removeAllListeners();
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
            const normalizedDelta = normalizer.processDelta(delta);
            let clientContent = '';

            if (SHOW_REASONING && inlineReasoning) {
              if (normalizedDelta.reasoning && !reasoningOpen) {
                clientContent += `<thinking>\n${normalizedDelta.reasoning}`;
                reasoningOpen = true;
              } else if (normalizedDelta.reasoning) {
                clientContent += normalizedDelta.reasoning;
              }

              if (normalizedDelta.content && reasoningOpen) {
                clientContent += `\n</thinking>\n\n${normalizedDelta.content}`;
                reasoningOpen = false;
              } else if (normalizedDelta.content) {
                clientContent += normalizedDelta.content;
              }
            } else {
              clientContent = normalizedDelta.content || '';
            }

            const { content: recoveredContent, toolCallDeltas } = toolRecovery.process(clientContent);
            clientContent = recoveredContent;
            if (toolCallDeltas.length > 0) {
              delta.tool_calls = toolCallDeltas;
              if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
            }

            delta.content = clientContent;

            if (SHOW_REASONING && normalizedDelta.reasoning) {
              delta.reasoning = normalizedDelta.reasoning;
              delta.reasoning_content = normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Upstream sent malformed chunk', type: 'stream_parse_error' } })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);
        if (buffer.length > MAX_BUFFER_SIZE) {
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Stream buffer overflow', type: 'stream_error' } })}\n\n`);
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) processLine(line);
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) processLine(line);
        }

        const flushedDelta = normalizer.flush();
        const toolRecoveryLeftover = toolRecovery.flush();
        if (toolRecoveryLeftover) {
          flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
        }

        if (flushedDelta.content || flushedDelta.reasoning) {
          const finalChunk = { choices: [{ delta: { content: flushedDelta.content || '' } }] };
          safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
        }

        if (!doneSent) safeWrite(res, 'data: [DONE]\n\n');
        streamEndedCleanly = true;
        if (!res.writableEnded) res.end();
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({ error: { message: 'Stream interrupted', type: 'stream_error' } })}\n\n`);
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
        model: usedModel,
        created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          const { content: cleanedContent, toolCalls: recoveredToolCalls } = extractLeakedToolCalls(content);
          content = cleanedContent;

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };
          if (recoveredToolCalls.length > 0) {
            finalMessage.tool_calls = [...(normalizedChoice.message?.tool_calls || []), ...recoveredToolCalls];
            if (!finalMessage.content || !finalMessage.content.trim()) finalMessage.content = null;
          }

          return {
            ...normalizedChoice,
            index: i,
            message: finalMessage,
            ...(recoveredToolCalls.length > 0 && { finish_reason: 'tool_calls' })
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.message, type: 'proxy_error', code: error.response?.status || 500 }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({ error: { message: error.message, type: 'proxy_error' } })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }
    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

// ─── Startup ──────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Fiel-Fast Hybrid proxy running on port ${PORT}`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
