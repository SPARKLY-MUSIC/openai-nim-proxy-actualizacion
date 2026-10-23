// server.js — Robust Hybrid OpenAI ↔ NIM Proxy (Fast, Stable & Prompt-Compliant)
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
const REQUEST_TIMEOUT_MS = 60000; // Timeout optimizado para velocidad
const VALIDATION_TIMEOUT_MS = 10000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

// Agent de red para reutilizar conexiones y acelerar la respuesta drásticamente
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping (Modelos rápidos y estables que respetan contexto) ────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-3.5': 'meta/llama-3.1-8b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-70b-instruct',
  'gpt-4o': 'meta/llama-3.3-70b-instruct',
  'claude-3-opus': 'meta/llama-3.1-70b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-8b-instruct',
  'gemini-pro': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-3.5o': 'meta/llama-3.1-8b-instruct',
  'gpt-4-flash': 'meta/llama-3.1-8b-instruct',
  'mistral': 'meta/llama-3.1-70b-instruct',
  'mistral-turbo': 'meta/llama-3.1-8b-instruct',
  'google-light': 'google/gemma-4-31b-it'
};

const FALLBACK_MODELS = [
  'meta/llama-3.1-8b-instruct',
  'meta/llama-3.3-70b-instruct',
  'google/gemma-4-31b-it'
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
      error: { message: 'Forbidden: Invalid or missing authentication', type: 'authentication_error', code: 403 }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: { message: 'Forbidden: Invalid authentication credentials', type: 'authentication_error', code: 403 }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) return;
  try {
    await nim.get('/models', {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` },
      timeout: VALIDATION_TIMEOUT_MS
    });
  } catch (err) {
    console.warn(`[VALIDATION] Skipped: ${err.message}`);
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

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
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
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.2.0-fast-compliant' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    const primaryModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];

    // Payload limpio y directo para que el modelo lea el historial y prompts sin alteraciones
    const baseRequest = {
      messages,
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
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) upstreamStream.removeAllListeners();
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
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

          if (delta && delta.reasoning_content) {
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
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => ({
          index: i,
          message: {
            role: choice.message?.role || 'assistant',
            content: choice.message?.content || '',
            tool_calls: choice.message?.tool_calls
          },
          finish_reason: choice.finish_reason || 'stop'
        })),
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
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`[PROXY] Fast & compliant proxy running on port ${PORT}`);
  validateModels().catch(() => {});
});
