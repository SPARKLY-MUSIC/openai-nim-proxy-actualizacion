// server.js — Stable & Fast OpenAI ↔ NIM Proxy
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

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 60000; 

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
  if (!NIM_API_KEY) {
    console.error('[FATAL] NIM_API_KEY is required.');
    process.exit(1);
  }
}
validateConfig();

// ─── Model Mapping (Modelo único ultra-estable para evitar errores 500) ─────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.1-8b-instruct',
  'gpt-4o': 'meta/llama-3.1-8b-instruct',
  'claude-3-opus': 'meta/llama-3.1-8b-instruct',
  'gemini-pro': 'meta/llama-3.1-8b-instruct'
};

const TARGET_MODEL = 'meta/llama-3.1-8b-instruct';

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
  if (!token || !CLIENT_AUTH_KEY || !safeTimingEqual(token, CLIENT_AUTH_KEY)) {
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

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.3.0-bulletproof' });
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
  let upstreamStream = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const baseRequest = {
      model: MODEL_MAPPING[model] || TARGET_MODEL,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const response = await nim.post('/chat/completions', baseRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT_MS
    });

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
          return;
        }
        try {
          const data = JSON.parse(line.slice(6));
          if (data.choices?.[0]?.delta?.reasoning_content) {
            delete data.choices[0].delta.reasoning_content;
          }
          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch {}
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) processLine(line);
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
    console.error('[PROXY] Error:', error.response?.data || error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.response?.data?.error?.message || error.message, type: 'api_error', code: error.response?.status || 500 }
      });
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
  console.log(`[PROXY] Running on port ${PORT}`);
});
