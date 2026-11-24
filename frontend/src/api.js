export async function requestQuestion(question, topK = 2, { fetchImpl = fetch, signal, documentId, documentIds } = {}) {
  if (typeof question !== 'string' || !question.trim()) throw new Error('请输入问题。')
  if (question.trim().length > 2000) throw new Error('问题不能超过 2000 个字符。')
  if (!Number.isInteger(topK) || topK < 1 || topK > 10) throw new Error('参考片段数量必须为 1 到 10。')
  const response = await fetchImpl('/api/query', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question.trim(), top_k: topK, ...(documentIds ? { document_ids: documentIds } : documentId ? { document_id: documentId } : {}) }), signal,
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

// FormData 会让浏览器自动生成包含 boundary 的 Content-Type。
export async function uploadDocument(file, { fetchImpl = fetch, signal, replaceId } = {}) {
  if (!(file instanceof Blob) || typeof file.name !== 'string') throw new Error('请先选择一个 TXT、PDF 或 Word (.docx) 文件。')
  if (!/\.(txt|pdf|docx)$/i.test(file.name)) throw new Error('目前只支持 .txt、.pdf 和 .docx 文件。')
  if (file.size === 0) throw new Error('不能上传空文件。')
  if (file.size > 1024 * 1024) throw new Error('文件不能超过 1 MiB。')
  const form = new FormData()
  form.append('file', file)
  const url = replaceId ? '/api/documents/' + encodeURIComponent(replaceId) : '/api/upload'
  const response = await fetchImpl(url, { method: replaceId ? 'PUT' : 'POST', body: form, signal })
  let data
  try { data = await response.json() } catch { throw new Error('服务返回了无法识别的内容，请稍后重试。') }
  if (!response.ok || data?.success !== true) throw new Error(data?.error || '文件上传失败，请稍后重试。')
  if (typeof data.document?.id !== 'string' || typeof data.document?.filename !== 'string' || data.document?.indexed !== false) {
    throw new Error('服务没有返回有效的上传结果。')
  }
  return data.document
}

export async function fetchDocuments({ fetchImpl = fetch, signal } = {}) {
  const response = await fetchImpl('/api/documents', { signal })
  let data
  try { data = await response.json() } catch { throw new Error('服务返回了无法识别的文档列表。') }
  if (!response.ok || data?.success !== true) throw new Error(data?.error || '文档列表加载失败，请稍后重试。')
  if (!Array.isArray(data.documents) || data.documents.some(document =>
    !document || !/^[0-9a-f]{32}$/.test(document.id) || typeof document.filename !== 'string' ||
    !Number.isInteger(document.size_bytes) || document.size_bytes < 0)) {
    throw new Error('服务没有返回有效的文档列表。')
  }
  return data.documents
}

export async function removeDocument(id, { signal } = {}) {
  const response = await fetch('/api/documents/' + encodeURIComponent(id), { method: 'DELETE', signal })
  const data = await response.json()
  if (!response.ok || data?.success !== true) throw new Error(data?.error || '移除资料失败，请稍后重试。')
}
