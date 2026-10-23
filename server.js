// server.js — Robust Hybrid OpenAI ↔ NIM Proxy (Optimized for Speed)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
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
const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────
function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests might be rejected if your frontend sends a key.');
  }
}
validateConfig();

// ─── Model Mapping (Optimizado y Rápido) ───────────────────────────────────
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-nano-8b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-sonnet': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek': 'deepseek-ai/deepseek-v3.1',
  'llama-70b': 'meta/llama-3.1-70b-instruct'
};

const FALLBACK_MODELS = [
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'deepseek-ai/deepseek-v3.1'
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
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
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
      console.warn(`[FALLBACK] Model failed: ${model}`, err.response?.status, err.message);
    }
  }
  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.2.0-fast' });
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
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const primaryModel = MODEL_MAPPING[model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];
    
    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let doneSent = false;

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
            delete delta.reasoning_content; // Evita bloqueos de buffer por razonamiento oculto pesado
          }
          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch (parseErr) {
          // Ignorar chunks corruptos menores para mantener velocidad
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        if (!doneSent) safeWrite(res, 'data: [DONE]\n\n');
        if (!res.writableEnded) res.end();
      });

      upstreamStream.on('error', () => {
        if (!res.writableEnded) res.end();
      });

      req.on('close', () => {
        if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
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
            content: choice.message?.content || ''
          },
          finish_reason: choice.finish_reason || 'stop'
        })),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('[PROXY] Error:', error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.message, type: 'invalid_request_error', code: error.response?.status || 500 }
      });
    }
  }
});

app.use((req, res) => {
  res.status(404).json({ error: { message: `Endpoint not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`[PROXY] Fast Hybrid proxy running on port ${PORT}`);
});
