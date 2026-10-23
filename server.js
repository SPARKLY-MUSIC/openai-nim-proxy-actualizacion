// server.js — Fiel Optimizado: Velocidad de Rápido + Respeto absoluto al usuario
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https'); // <-- NUEVO: Para keepAlive
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuración ───────────────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 120000; // 120s balanceado
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

// ⚡ OPTIMIZACIÓN 1: Keep-Alive Agent (Reutiliza conexiones, reduce latencia drásticamente)
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

// ⚡ OPTIMIZACIÓN 2: Instancia Axios dedicada con el agente
const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Validación de Config ──────────────────────────────────────────────────────
function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required.');
  if (!CLIENT_AUTH_KEY) console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
}
validateConfig();

// ─── Mapeo de Modelos (Versión Ultra-Rápida y Fiel) ────────────────────────
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
  'mistral': 'meta/llama-3.1-70b-instruct',
  'mistral-turbo': 'meta/llama-3.1-8b-instruct',
  'google-light': 'google/gemma-4-31b-it'
};

const FALLBACK_MODELS = [
  'meta/llama-3.1-8b-instruct',
  'meta/llama-3.3-70b-instruct',
  'google/gemma-4-31b-it'
];


const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Middleware ─────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Subido a 50mb para soportar prompts largos/historial

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try { return timingSafeEqual(Buffer.from(a), Buffer.from(b)); } 
  catch { return false; }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') return next();
  const token = extractBearerToken(req.headers.authorization);
  
  if (!token || !CLIENT_AUTH_KEY || !safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({ error: { message: 'Forbidden: Invalid authentication', type: 'authentication_error', code: 403 } });
  }
  next();
});

// ─── Validación de Modelos ──────────────────────────────────────────────────
async function validateModels() {
  if (SKIP_VALIDATION) return;
  console.log('[VALIDATION] Checking model availability...');
  try {
    const response = await nim.get('/models', { // <-- Usa la instancia optimizada
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      timeout: VALIDATION_TIMEOUT_MS
    });
    const availableModels = new Set((response.data.data || []).map(m => m.id));
    const invalid = [];
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (!availableModels.has(nimId)) invalid.push({ alias, nimId, error: 'Not in catalog' });
    }
    if (invalid.length > 0 && DISCORD_WEBHOOK_URL) {
      await axios.post(DISCORD_WEBHOOK_URL, {
        embeds: [{ title: '⚠️ Model Validation Failed', description: `${invalid.length} models invalid`, color: 0xff4444 }]
      }, { timeout: 5000 });
    }
  } catch (err) {
    console.warn('[VALIDATION] Skipped due to error:', err.message);
  }
}

// ─── Helpers: Safe Stream & Cooldowns ───────────────────────────────────────
function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) { console.warn('[STREAM] Write failed:', err.message); }
  return false;
}

// ⚡ OPTIMIZACIÓN 3: Cooldowns para evitar esperar a modelos saturados (429/403)
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

// ─── Fallback Chain Optimizado ──────────────────────────────────────────────
async function callWithFallback(baseRequest, models) {
  let lastError = null;
  
  // Filtra modelos en cooldown primero
  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    try {
      const res = await nim.post('/chat/completions', { ...baseRequest, model }, {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        responseType: baseRequest.stream ? 'stream' : 'json',
        timeout: REQUEST_TIMEOUT_MS
      });
      return { response: res, model };
    } catch (err) {
      lastError = err;
      const status = err.response?.status;
      console.warn(`[FALLBACK] Model failed: ${model} | Status: ${status}`);
      
      if (status === 401) throw err; // Error de API key, no tiene sentido seguir
      if (status === 403) setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      if (status === 429) setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
    }
  }
  throw lastError || new Error('All models failed');
}

// ─── Rutas ──────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => res.json({ status: 'ok', version: '3.0.0-fiel-optimizado' }));

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nim-proxy-fiel'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    // ⚡ OPTIMIZACIÓN 4: Preservar TODOS los campos del usuario (tools, stop, top_p, etc.)
    // pero asegurando que los valores por defecto no rompan la solicitud.
    const { model: _droppedModel, ...forwardedFields } = req.body;
    
    // Garantizamos que 'messages' exista, que es lo que respeta el prompt
    if (!forwardedFields.messages || !Array.isArray(forwardedFields.messages)) {
      return res.status(400).json({ error: { message: 'Missing or invalid "messages" array', type: 'invalid_request_error' } });
    }

    const primaryModel = MODEL_MAPPING[req.body.model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])]; // Elimina duplicados

    const baseRequest = {
      ...forwardedFields, // <-- Aquí van intactos: messages, tools, tool_choice, stop, temperature, etc.
      temperature: req.body.temperature ?? 0.7,
      max_tokens: Math.min(req.body.max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: req.body.stream || false,
      ...(ENABLE_THINKING_MODE ? { extra_body: { chat_template_kwargs: { thinking: true } } } : {})
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (req.body.stream) {
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
        if (upstreamStream) upstreamStream.removeAllListeners();
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;
        if (line.includes('[DONE]')) {
          if (!doneSent) { safeWrite(res, 'data: [DONE]\n\n'); doneSent = true; }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING && reasoning) {
              if (!reasoningOpen) {
                content = `<thinking>\n${reasoning}`;
                reasoningOpen = true;
              } else {
                content = reasoning;
              }
              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }
            delta.content = content;
            delete delta.reasoning_content; // Limpieza para no duplicar
          }
          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON chunk');
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);
        if (buffer.length > MAX_BUFFER_SIZE) {
          safeWrite(res, 'data: {"error":{"message":"Buffer overflow"}}\n\ndata: [DONE]\n\n');
          res.end(); upstreamStream.destroy(); cleanup(); return;
        }
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) processLine(line);
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) { for (const line of buffer.split('\n')) processLine(line); }
        if (!doneSent) safeWrite(res, 'data: [DONE]\n\n');
        streamEndedCleanly = true;
        if (!res.writableEnded) res.end();
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, `data: {"error":{"message":"Stream interrupted"}}\n\ndata: [DONE]\n\n`);
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
      // Respuesta No-Streaming (Mantiene la fidelidad del mensaje original)
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: usedModel, // Usar el modelo real que respondió
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
              tool_calls: choice.message?.tool_calls // <-- Se preservan las herramientas
            },
            finish_reason: choice.finish_reason || 'stop'
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
      safeWrite(res, `data: {"error":{"message":"${error.message}"}}\n\ndata: [DONE]\n\n`);
      res.end();
    }
    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  }
});

app.use((req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

// ─── Startup ───────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[PROXY] Fiel-Optimizado corriendo en puerto ${PORT}`);
  console.log(`[PROXY] Keep-Alive: ACTIVO | Max Tokens: ${MAX_TOKENS_LIMIT}`);
  validateModels().catch(err => console.error('[VALIDATION] Startup check failed:', err.message));
});
