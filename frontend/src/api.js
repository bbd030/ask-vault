export async function requestQuestion(question, topK = 2, { fetchImpl = fetch, signal } = {}) {
  if (typeof question !== 'string' || !question.trim()) throw new Error('请输入问题。')
  if (question.trim().length > 2000) throw new Error('问题不能超过 2000 个字符。')
  if (!Number.isInteger(topK) || topK < 1 || topK > 10) throw new Error('参考片段数量必须为 1 到 10。')
  const response = await fetchImpl('/api/query', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question.trim(), top_k: topK }), signal,
  })
  let data
  try { data = await response.json() } catch { throw new Error('服务返回了无法识别的内容，请稍后重试。') }
  if (!response.ok || data?.success !== true) throw new Error(data?.error || '问答暂时失败，请稍后重试。')
  if (typeof data.answer !== 'string' || !data.answer.trim()) throw new Error('服务没有返回有效回答。')
  return { ...data, sources: Array.isArray(data.sources) ? data.sources : [], matches: Array.isArray(data.matches) ? data.matches : [] }
}

export async function checkHealth({ fetchImpl = fetch } = {}) {
  const response = await fetchImpl('/api/health')
  const data = await response.json()
  return response.ok && data?.status === 'ok' && data?.service === 'AskVault'
}
