// server.js - OpenAI to NVIDIA NIM API Proxy (Optimized for Janitor AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = process.env.SHOW_REASONING === 'true' || false;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true' || false;

// 🎯 OPTIMIZED MODEL MAPPING FOR JANITOR AI
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
  'gemini-pro-vision': 'nvidia/nemotron-nano-12b-v2-vl',
  'gpt-4-reasoning': 'moonshotai/kimi-k2-instruct-1113',
  'deepseek': 'deepseek-ai/deepseek-v3.1',
  'llama-70b': 'meta/llama-3.1-70b-instruct',
  'llama-405b': 'meta/llama-3.1-405b-instruct',
  'llama-8b': 'meta/llama-3.1-8b-instruct'
};

// 🛡️ ROLEPLAY GUARD
const RP_GUARD_INSTRUCTION = `You are ONLY the character described in the system prompt or conversation. Follow these rules strictly:
- You ONLY speak, act, and think as the character. You do NEVER write or generate any dialogue, actions, or thoughts for the user or any other character that the user is playing.
- Do NOT use labels like "User:", "Human:", "You:" or any prefix to simulate the user's side of the conversation.
- Do NOT continue the conversation by inventing what the user says or does next.
- Stop your response immediately after your character's turn ends.
- If you feel the scene needs a reaction from the user, end your response and wait.`;

// 🛡️ ROLEPLAY GUARD - Strips any text where the model broke character
function stripUserBreakout(text) {
  const lines = text.split('\n');
  const cleaned = [];
  let dropping = false;
  const userLabels = [
    /^(User|Human|You|Me|Player)\s*[:：]/i,
    /^---+\s*$/,
    /^\*{0,3}\s*(User|Human|You|Me|Player)\s*\*{0,3}\s*[:：]/i
  ];

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

  const result = cleaned.join('\n');
  const lastUserLabel = result.search(/\n(?:User|Human|You|Me|Player)\s*[:：]/i);
  if (lastUserLabel !== -1) {
    return result.substring(0, lastUserLabel).trimEnd();
  }
  return result.trimEnd();
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy (Janitor AI Optimized)', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_configured: !!NIM_API_KEY
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy',
    nim_model: MODEL_MAPPING[model]
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NIM_API_KEY not configured.', type: 'configuration_error', code: 500 }
      });
    }

    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 1. SELECCIÓN DE MODELO INSTANTÁNEA (SIN PETICIONES DE PRUEBA)
    let nimModel = MODEL_MAPPING[model];
    
    if (!nimModel) {
      const modelLower = model.toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('opus')) {
        nimModel = 'deepseek-ai/deepseek-v3.2';
      } else if (modelLower.includes('deepseek')) {
        nimModel = 'deepseek-ai/deepseek-v3.1';
      } else if (modelLower.includes('claude-sonnet') || modelLower.includes('70b')) {
        nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
      } else if (modelLower.includes('3.5') || modelLower.includes('haiku') || modelLower.includes('fast')) {
        nimModel = 'nvidia/llama-3.1-nemotron-nano-8b-v1';
      } else if (modelLower.includes('gemini') || modelLower.includes('qwen')) {
        nimModel = 'qwen/qwen3-next-80b-a3b-thinking';
      } else {
        nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1.5'; // Fallback por defecto
      }
    }
    
    // 2. INYECTAR SYSTEM PROMPT (Tu lógica original que SÍ funciona)
    const systemIndex = messages.findIndex(m => m.role === 'system');
    if (systemIndex !== -1) {
      messages[systemIndex] = {
        ...messages[systemIndex],
        content: messages[systemIndex].content + '\n\n' + RP_GUARD_INSTRUCTION
      };
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

    // 3. PETICIÓN DIRECTA A NIM (Sin demoras)
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 55000 // Timeout de seguridad para evitar cuelgues
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      let contentAccumulator = '';
      let flushedUpTo = 0;
      const LOOKAHEAD = 200;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              if (contentAccumulator.length > flushedUpTo) {
                const remaining = stripUserBreakout(contentAccumulator.substring(flushedUpTo));
                if (remaining.length > 0) {
                  res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: remaining }, index: 0 }] })}\n\n`);
                }
              }
              res.write('data: [DONE]\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content || '';
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  data.choices[0].delta.content = content;
                  delete data.choices[0].delta.reasoning_content;
                }

                const chunkText = data.choices[0].delta.content || '';
                if (chunkText) {
                  contentAccumulator += chunkText;
                  const filtered = stripUserBreakout(contentAccumulator);
                  const safeEnd = Math.max(flushedUpTo, filtered.length - LOOKAHEAD);
                  if (safeEnd > flushedUpTo) {
                    const toSend = filtered.substring(flushedUpTo, safeEnd);
                    flushedUpTo = safeEnd;
                    data.choices[0].delta.content = toSend;
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                  }
                  return;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Ignorar líneas de stream no JSON
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
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
    let errorMessage = error.message || 'Internal server error';
    if (error.response?.status === 401) errorMessage = 'Invalid NVIDIA API key.';
    else if (error.response?.status === 429) errorMessage = 'Rate limit exceeded.';
    else if (error.response?.data?.detail) errorMessage = error.response.data.detail;
    
    res.status(error.response?.status || 500).json({
      error: { message: errorMessage, type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found.`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🚀 OpenAI → NVIDIA NIM Proxy (Janitor AI Optimized)');
  console.log(`📡 Server running on port ${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log('═══════════════════════════════════════════════════════');
});
