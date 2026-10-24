async function callWithFallback(baseRequest, models) {
  let lastError = null;

  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    try {
      // Limpiamos extra_body si algún modelo no soporta thinking para evitar conflictos
      const requestPayload = { ...baseRequest, model };
      
      const res = await nim.post(
        '/chat/completions',
        requestPayload,
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
      const errorMsg = err.response?.data?.error?.message || err.message;

      console.warn(`[FALLBACK] Model failed: ${model} (Status: ${status}) - Message: ${errorMsg}`);

      if (status === 401) throw err;
      if (status === 403 || status === 404 || status === 410) setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      if (status === 429) setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
    }
  }

  throw lastError || new Error('All models failed');
}
