import test from 'node:test'
import assert from 'node:assert/strict'
import { fetchDocuments } from '../src/api.js'

test('document list preserves distinct IDs for same-name documents', async () => {
  const docs = ['a', 'b'].map(id => ({ id: id.repeat(32), filename: '同名.txt', size_bytes: 123 }))
  const signal = new AbortController().signal
  const actual = await fetchDocuments({ signal, fetchImpl: async (url, options) => {
    assert.equal(url, '/api/documents')
    assert.equal(options.signal, signal)
    return { ok: true, json: async () => ({ success: true, documents: docs }) }
  } })
  assert.deepEqual(actual, docs)
})

test('empty document list is a valid result', async () => {
  assert.deepEqual(await fetchDocuments({ fetchImpl: async () => ({ ok: true, json: async () => ({ success: true, documents: [] }) }) }), [])
})

test('malformed lists and API failures are reported instead of looking empty', async () => {
  for (const documents of [null, {}, [null], [{ id: '../x', filename: 'x', size_bytes: 1 }], [{ id: 'a'.repeat(32), filename: 3, size_bytes: 1 }]]) {
    await assert.rejects(fetchDocuments({ fetchImpl: async () => ({ ok: true, json: async () => ({ success: true, documents }) }) }), /有效的文档列表/)
  }
  await assert.rejects(fetchDocuments({ fetchImpl: async () => ({ ok: false, json: async () => ({ success: false, error: '列表读取失败' }) }) }), /列表读取失败/)
  await assert.rejects(fetchDocuments({ fetchImpl: async () => ({ ok: true, json: async () => { throw new SyntaxError() } }) }), /无法识别/)
})
