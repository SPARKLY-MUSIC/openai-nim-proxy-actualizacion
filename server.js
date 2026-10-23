// server.js - OpenAI to NVIDIA NIM API Proxy (Optimized for Janitor AI & Vercel)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING === 'true' || false;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true' || false;

const MODEL_MAPPING = {
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',
  'claude-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-sonnet': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-nano-8b-v1',
  'gpt-3.5-turbo-16k': 'nvidia/nvidia-nemotron-nano-9b-v2',
  'claude-haiku': 'nvidia/nemotron-3-nano-30b-a3b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'llama-70b': 'meta/llama-3.1-70b-instruct',
  'llama-405b': 'meta/llama-3.1-405b-instruct',
  'llama-8b': 'meta/llama-3.1-8b-instruct'
};

const RP_GUARD_INSTRUCTION = `You are ONLY the character described in the system prompt or conversation. Follow these rules strictly:
- You ONLY speak, act, and think as the character. You do NEVER write or generate any dialogue, actions, or thoughts for the user.
- Do NOT use labels like "User:", "Human:", "You:" to simulate the user's side.
- Stop your response immediately after your character's turn ends.`;

function stripUserBreakout(text) {
  const lines = text.split('\n');
  const cleaned = [];
  let dropping = false;
  const userLabels = [/^(User|Human|You|Me|Player)\s*[:：]/i, /^---+\s*$/, /^\*{0,3}\s*(User|Human|You|Me|Player)\s*\*{0,3}\s*[:：]/i];

  for (const line of lines) {
    const trimmed = line.trim();
    if (userLabels.some(pattern => pattern.test(trimmed))) {
      dropping = true;
      continue;
    }
    if (dropping) {
      if (trimmed === '') continue;
      if (trimmed.startsWith('*')) {
        dropping = false;
        cleaned.push(line);
      }
      continue;
    }
    cleaned.push(line);
  }
  return cleaned.join('\n').trimEnd();
}

app.get('/health', (req, res) => res.json({ status: 'ok', nim_api_configured: !!NIM_API_KEY }));

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy', nim_model: MODEL_MAPPING[model]
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({ error: { message: 'NIM_API_KEY not configured.', type: 'configuration_error', code: 500 } });
    }

    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 1. SELECCIÓN DE MODELO INMEDIATA (SIN PETICIONES DE PRUEBA)
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const modelLower = model.toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('opus')) nimModel = 'deepseek-ai/deepseek-v3.2';
      else if (modelLower.includes('deepseek')) nimModel = 'deepseek-ai/deepseek-v3.1';
      else if (modelLower.includes('claude-sonnet') || modelLower.includes('70b')) nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
      else if (modelLower.includes('3.5') || modelLower.includes('haiku') || modelLower.includes('fast')) nimModel = 'nvidia/llama-3.1-nemotron-nano-8b-v1';
      else nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1.5'; // Fallback seguro y rápido
    }

    // 2. INYECTAR SYSTEM PROMPT (Esto es lo que hace que cumpla tu prompt)
    const systemIndex = messages.findIndex(m => m.role === 'system');
    if (systemIndex !== -1) {
      messages[systemIndex] = { ...messages[systemIndex], content: messages[systemIndex].content + '\n\n' + RP_GUARD_INSTRUCTION };
    } else {
      messages.unshift({ role: 'system', content: RP_GUARD_INSTRUCTION });
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    // 3. PETICIÓN DIRECTA A NIM (Con timeout controlado para no colgar Vercel)
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      timeout: 50000 // 50 segundos (deja margen antes del límite de Vercel)
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.on('data', (chunk) => {
        const lines = chunk.toString().split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write('data: [DONE]\n\n');
              continue;
            }
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const content = data.choices[0].delta.content || '';
                const reasoning = data.choices[0].delta.reasoning_content;
                
                if (SHOW_REASONING && reasoning) {
                   data.choices[0].delta.content = `<think>\n${reasoning}\n</think>\n${content}`;
                   delete data.choices[0].delta.reasoning_content;
                } else {
                   delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Ignorar líneas que no son JSON válido (comunes en streams)
            }
          }
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Respuesta no streaming (aquí sí aplicamos el filtro completo de forma segura)
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          fullContent = stripUserBreakout(fullContent); 
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.detail || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Proxy corriendo en puerto ${PORT}`);
});
