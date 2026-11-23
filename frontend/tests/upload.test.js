import test from 'node:test'
import assert from 'node:assert/strict'
import { uploadDocument } from '../src/api.js'

const document = { id: 'test-id', filename: '制度.txt', indexed: false, size_bytes: 12 }

test('upload sends the original file via FormData with browser-managed headers', async () => {
  const file = new File(['报销需要发票。'], '制度.txt', { type: 'text/plain' })
  const signal = new AbortController().signal
  let captured
  const result = await uploadDocument(file, { signal, fetchImpl: async (...args) => {
    captured = args
    return { ok: true, json: async () => ({ success: true, document }) }
  } })
  assert.equal(captured[0], '/api/upload')
  assert.equal(captured[1].method, 'POST')
  assert.equal(captured[1].headers, undefined)
  assert.equal(captured[1].signal, signal)
  assert.equal(captured[1].body.get('file').name, '制度.txt')
  assert.equal(await captured[1].body.get('file').text(), '报销需要发票。')
  assert.deepEqual(result, document)
})

test('invalid file selection is rejected before any network request', async () => {
  let calls = 0
  const fetchImpl = async () => { calls++; throw new Error('unexpected fetch') }
  for (const file of [null, {}, new File(['hello'], 'a.pdf'), new File([], 'a.txt'), new File(['x'.repeat(1024 * 1024 + 1)], 'a.txt')]) {
    await assert.rejects(uploadDocument(file, { fetchImpl }))
  }
  assert.equal(calls, 0)
})

test('uppercase TXT and exact size boundary can be uploaded', async () => {
  const file = new File(['x'.repeat(1024 * 1024)], 'A.TXT')
  await uploadDocument(file, { fetchImpl: async () => ({ ok: true, json: async () => ({ success: true, document }) }) })
})

test('backend validation errors and malformed responses reach the caller', async () => {
  const file = new File(['hello'], 'a.txt')
  await assert.rejects(uploadDocument(file, { fetchImpl: async () => ({ ok: false, json: async () => ({ success: false, error: 'TXT 文件必须使用 UTF-8 编码' }) }) }), /UTF-8/)
  await assert.rejects(uploadDocument(file, { fetchImpl: async () => ({ ok: true, json: async () => { throw new SyntaxError() } }) }), /无法识别/)
  await assert.rejects(uploadDocument(file, { fetchImpl: async () => ({ ok: true, json: async () => ({ success: true }) }) }), /有效的上传结果/)
})

test('network and abort failures propagate so the page can offer retry', async () => {
  const file = new File(['hello'], 'a.txt')
  const aborted = new DOMException('aborted', 'AbortError')
  for (const error of [new TypeError('offline'), aborted]) {
    await assert.rejects(uploadDocument(file, { fetchImpl: async () => { throw error } }), actual => actual === error)
  }
})
