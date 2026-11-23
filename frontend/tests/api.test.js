import test from 'node:test'
import assert from 'node:assert/strict'
import { requestQuestion, checkHealth } from '../src/api.js'

const success = { success: true, answer: '需要发票。[1]', sources: [{ id: 1, source: 'sample.txt', chunk_id: 2 }], matches: [{ score: 0.9, source: 'sample.txt', chunk_id: 2, text: '需要发票。' }] }

test('question request preserves source data and sends the selected top_k', async () => {
  let recorded
  const signal = new AbortController().signal
  const result = await requestQuestion('  报销需要哪些材料？  ', 3, { signal, fetchImpl: async (...args) => { recorded = args; return { ok: true, json: async () => success } } })
  assert.equal(recorded[0], '/api/query')
  assert.deepEqual(JSON.parse(recorded[1].body), { question: '报销需要哪些材料？', top_k: 3 })
  assert.equal(recorded[1].headers['Content-Type'], 'application/json')
  assert.equal(recorded[1].signal, signal)
  assert.equal(result.sources[0].chunk_id, 2)
})

test('invalid questions and top_k are rejected before any network request', async () => {
  let calls = 0
  const fetchImpl = async () => { calls++; throw new Error('unexpected fetch') }
  for (const question of ['', ' ', null, 12, '问'.repeat(2001)]) await assert.rejects(requestQuestion(question, 2, { fetchImpl }))
  for (const count of [0, 11, 1.5, true]) await assert.rejects(requestQuestion('材料？', count, { fetchImpl }))
  assert.equal(calls, 0)
})

test('API errors are surfaced for the error and retry interface', async () => {
  await assert.rejects(requestQuestion('材料？', 2, { fetchImpl: async () => ({ ok: false, json: async () => ({ success: false, error: '问答服务暂不可用' }) }) }), /问答服务暂不可用/)
})

test('invalid JSON and missing answers are handled', async () => {
  await assert.rejects(requestQuestion('材料？', 2, { fetchImpl: async () => ({ ok: true, json: async () => { throw new SyntaxError('invalid JSON') } }) }), /无法识别/)
  await assert.rejects(requestQuestion('材料？', 2, { fetchImpl: async () => ({ ok: true, json: async () => ({ success: true }) }) }), /有效回答/)
})

test('missing optional evidence lists become empty arrays', async () => {
  const result = await requestQuestion('材料？', 2, { fetchImpl: async () => ({ ok: true, json: async () => ({ success: true, answer: '未找到足够依据。' }) }) })
  assert.deepEqual(result.sources, [])
  assert.deepEqual(result.matches, [])
})

test('health checks require an actual AskVault response', async () => {
  assert.equal(await checkHealth({ fetchImpl: async url => { assert.equal(url, '/api/health'); return { ok: true, json: async () => ({ status: 'ok', service: 'AskVault' }) } } }), true)
  assert.equal(await checkHealth({ fetchImpl: async () => ({ ok: false, json: async () => ({ status: 'ok' }) }) }), false)
})
