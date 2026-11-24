<script setup>
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { checkHealth, requestQuestion, uploadDocument, fetchDocuments } from './api.js'

const documents = ref([])
const documentsLoading = ref(false)
const documentsError = ref('')
const currentDocument = ref(null)

async function refreshDocuments() {
  if (documentsLoading.value || busy.value || uploading.value) return
  documentsLoading.value = true
  documentsError.value = ''
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 10000)
  try {
    documents.value = await fetchDocuments({ signal: controller.signal })
    if (currentDocument.value && !documents.value.some(document => document.id === currentDocument.value.id)) {
      currentDocument.value = null
      uploadedDocument.value = null
      newConversation()
    }
  } catch (error) {
    documentsError.value = error.name === 'AbortError' ? '文档列表加载超时，请重试。'
      : error instanceof TypeError ? '无法连接文档服务，请确认后端已启动。' : error.message
  } finally { clearTimeout(timer); documentsLoading.value = false }
}
function switchDocument(event) {
  if (busy.value || uploading.value || documentsLoading.value) return
  const id = event.target.value
  if (id === (currentDocument.value?.id || '')) return
  currentDocument.value = documents.value.find(document => document.id === id) || null
  uploadedDocument.value = null
  newConversation()
}
const selectedFile = ref(null)
const fileInput = ref(null)
const uploading = ref(false)
const uploadError = ref('')
const uploadedDocument = ref(null)

function selectFile(event) {
  selectedFile.value = event.target.files?.[0] || null
  uploadError.value = ''
  uploadedDocument.value = null
}
async function submitUpload() {
  if (uploading.value || busy.value || documentsLoading.value || !selectedFile.value) return
  uploading.value = true
  uploadError.value = ''
  uploadedDocument.value = null
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 30000)
  try {
    uploadedDocument.value = await uploadDocument(selectedFile.value, { signal: controller.signal })
    documents.value = [uploadedDocument.value, ...documents.value.filter(document => document.id !== uploadedDocument.value.id)]
    currentDocument.value = uploadedDocument.value
    newConversation()
    selectedFile.value = null
    if (fileInput.value) fileInput.value.value = ''
  } catch (error) {
    uploadError.value = error.name === 'AbortError' ? '上传等待超时，文件可能已保存，请稍后确认。'
      : error instanceof TypeError ? '暂时无法连接上传服务，请确认后端已启动。' : error.message
  } finally { clearTimeout(timer); uploading.value = false }
}

const question = ref('')
const topK = ref(2)
const messages = ref([])
const busy = ref(false)
const connection = ref('checking')
const selectedAnswerId = ref(null)
const selectedSource = ref('')
const chatEnd = ref(null)
const input = ref(null)
let sequence = 0
const examples = ['报销需要哪些材料？', '报销申请应在什么时候提交？', '超过提交日期会怎样处理？']
const activeAnswer = computed(() => messages.value.find(message => message.id === selectedAnswerId.value))
const connectionLabel = computed(() => ({ checking: '正在连接接口', online: '问答接口已连接', offline: '问答接口未连接' }[connection.value]))
const sourceKey = source => `${source.source}:${source.chunk_id}`
const referenceFor = match => activeAnswer.value?.sources.find(source => sourceKey(source) === sourceKey(match))
const scrollToEnd = async () => { await nextTick(); chatEnd.value?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }

async function refreshHealth() {
  connection.value = 'checking'
  try { connection.value = await checkHealth() ? 'online' : 'offline' } catch { connection.value = 'offline' }
}
function newConversation() {
  if (busy.value) return
  messages.value = []; question.value = ''; selectedAnswerId.value = null; selectedSource.value = ''
  input.value?.focus()
}
async function inspectSources(message, source) {
  selectedAnswerId.value = message.id
  selectedSource.value = source ? sourceKey(source) : ''
  await nextTick()
  if (window.matchMedia('(max-width: 900px)').matches) {
    document.querySelector('.sources-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}
async function runQuery(message) {
  busy.value = true; message.state = 'loading'; message.error = ''; message.answer = ''
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 90000)
  scrollToEnd()
  try {
    const result = await requestQuestion(message.question, message.topK, { signal: controller.signal, documentId: message.documentId })
    message.answer = result.answer; message.sources = result.sources; message.matches = result.matches; message.state = 'done'
    selectedAnswerId.value = message.id
    selectedSource.value = result.sources.length ? sourceKey(result.sources[0]) : ''
    connection.value = 'online'
  } catch (error) {
    message.state = 'error'
    refreshHealth()
    message.error = error.name === 'AbortError' ? '等待回答超时，请稍后重试。' : error instanceof TypeError ? '暂时无法连接问答服务，请确认后端已启动。' : error.message
  } finally { clearTimeout(timer); busy.value = false; scrollToEnd() }
}
async function submitQuestion() {
  const text = question.value.trim()
  if (!text || busy.value || uploading.value || documentsLoading.value) return
  messages.value.push({ id: ++sequence, role: 'user', text })
  const reply = reactive({ id: ++sequence, role: 'assistant', question: text, documentId: currentDocument.value?.id, topK: topK.value, state: 'loading', answer: '', sources: [], matches: [], error: '' })
  messages.value.push(reply); question.value = ''
  await runQuery(reply)
}
function handleKey(event) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitQuestion() }
}
function chooseExample(example) { question.value = example; input.value?.focus() }
onMounted(() => { refreshHealth(); refreshDocuments() })
</script>

<template>
  <div class="workspace">
    <aside class="sidebar">
      <a class="brand" href="/" aria-label="AskVault 首页"><span class="brand-mark">A</span><span>AskVault<small>让知识成为答案</small></span></a>
      <p class="nav-caption">工作空间</p>
      <div class="nav-item active"><span class="nav-icon">◈</span>知识问答</div>
      <section class="library-card"><span class="eyebrow">当前资料</span><div class="file-row"><span class="file-icon">{{ currentDocument?.filename.split('.').at(-1).toUpperCase() || 'TXT' }}</span><div><strong>{{ currentDocument?.filename || 'sample.txt' }}</strong><span>当前问答资料</span></div></div><p>围绕文档提问，查看回答与参考片段。</p></section>
      <section class="documents-card" aria-labelledby="documents-heading">
        <div class="documents-heading"><h2 id="documents-heading">文档列表</h2><button type="button" @click="refreshDocuments" :disabled="documentsLoading || uploading || busy">{{ documentsLoading ? '加载中…' : '刷新列表' }}</button></div>
        <label for="current-document">选择问答资料</label>
        <select id="current-document" :value="currentDocument?.id || ''" @change="switchDocument" :disabled="documentsLoading || uploading || busy">
          <option value="">sample.txt（默认资料）</option>
          <option v-for="document in documents" :key="document.id" :value="document.id">{{ document.filename }} · {{ document.id.slice(0, 6) }} · {{ (document.size_bytes / 1024).toFixed(1) }} KiB</option>
        </select>
        <p v-if="documentsLoading" role="status">正在读取已上传的资料…</p>
        <p v-else-if="documentsError" role="alert" class="upload-error">{{ documentsError }}</p>
        <p v-else-if="!documents.length">还没有上传资料，可先使用默认文档。</p>
        <p v-else>已上传 {{ documents.length }} 份资料。切换会清空当前对话，每次只查询所选文档。</p>
      </section>
      <section class="upload-card" aria-labelledby="upload-heading">
        <h2 id="upload-heading">上传资料</h2>
        <p class="upload-help" id="upload-help">支持 UTF-8 TXT、有文字层的 PDF 和 Word (.docx)，最大 1 MiB。</p>
        <form @submit.prevent="submitUpload" :aria-busy="uploading">
          <label for="document-file">选择文件</label>
          <input id="document-file" ref="fileInput" type="file" accept=".txt,.pdf,.docx,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" aria-describedby="upload-help" :disabled="uploading || busy || documentsLoading" @change="selectFile">
          <button class="upload-button" type="submit" :disabled="uploading || busy || documentsLoading || !selectedFile">{{ uploading ? '上传中…' : uploadError ? '重试上传' : '上传文件' }}</button>
        </form>
        <p v-if="uploading" class="upload-feedback" role="status">正在上传，请稍候…</p>
        <p v-if="uploadError" class="upload-feedback upload-error" role="alert">{{ uploadError }}</p>
        <div v-if="uploadedDocument" class="upload-feedback upload-success" role="status">
          <strong>{{ uploadedDocument.filename }}</strong>
          <span>已切换为当前资料，可开始提问。首次提问需要建立索引。</span>
        </div>
      </section>
      <div class="sidebar-note"><span class="note-line"></span><strong>你的知识，随时可查</strong><p>清晰的问题，配上可靠的资料。</p></div>
      <div class="sidebar-footer">AskVault · 本机工作台</div>
    </aside>

    <main class="main-area">
      <header class="topbar"><span>工作空间 <span class="breadcrumb">/ 知识问答</span></span><button class="connection" :class="connection" @click="refreshHealth" :disabled="connection === 'checking'"><span class="status-dot"></span>{{ connectionLabel }}</button></header>
      <section class="chat-room">
        <div class="chat-heading"><div><span class="eyebrow green">KNOWLEDGE CHAT</span><h1>向你的知识库提问</h1><p>结合文档内容回答，并展示参考来源。</p></div><button class="secondary-button" @click="newConversation" :disabled="busy">＋ 新建对话</button></div>
        <div class="chat-history" aria-live="polite" aria-busy="busy">
          <div v-if="!messages.length" class="welcome">
            <div class="welcome-icon"><svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M7 6h13l5 5v15H7V6Z" stroke="currentColor" stroke-width="1.7"/><path d="M19 6v6h6M11 17h10M11 21h7" stroke="currentColor" stroke-width="1.7"/></svg></div>
            <h2>答案，从你的资料开始</h2><p>{{ currentDocument ? '请围绕当前上传的资料提问。' : '试着询问报销材料、提交时间或具体规定。' }}</p>
            <div v-if="!currentDocument" class="suggestions"><button v-for="example in examples" :key="example" @click="chooseExample(example)">{{ example }}<span>试着提问</span></button></div>
          </div>
          <article v-for="message in messages" :key="message.id" class="message" :class="message.role">
            <template v-if="message.role === 'user'"><div class="user-bubble">{{ message.text }}</div></template>
            <template v-else>
              <div class="assistant-label"><span class="assistant-icon">A</span>AskVault<span class="answer-kind">文档问答</span></div>
              <div v-if="message.state === 'loading'" class="loading-answer" role="status"><span class="thinking-dots"><i></i><i></i><i></i></span>正在检索资料并生成回答…</div>
              <div v-else-if="message.state === 'error'" class="error-answer" role="alert"><p>{{ message.error }}</p><button class="text-button" @click="runQuery(message)" :disabled="busy || uploading || documentsLoading">重试这条问题</button></div>
              <template v-else><div class="answer-body">{{ message.answer }}</div><div v-if="message.sources.length" class="source-chips"><button v-for="source in message.sources" :key="source.id" @click="inspectSources(message, source)" :aria-label="`查看参考资料 ${source.id}`"><span>[{{ source.id }}]</span>{{ source.source }}<small>片段 {{ source.chunk_id }}</small></button></div><button v-if="message.matches.length" class="text-button view-sources" @click="inspectSources(message)">查看 {{ message.matches.length }} 个检索片段</button></template>
            </template>
          </article>
          <div ref="chatEnd"></div>
        </div>
        <form class="composer" @submit.prevent="submitQuestion()">
          <label class="sr-only" for="question">你的问题</label><textarea id="question" ref="input" v-model="question" placeholder="输入你想从文档中了解的问题…" rows="3" maxlength="2000" :disabled="busy" @keydown="handleKey"></textarea>
          <div class="composer-toolbar"><div class="retrieval-option"><label for="top-k">参考片段数量</label><select id="top-k" v-model.number="topK" :disabled="busy"><option v-for="count in [1, 2, 3, 5, 10]" :key="count" :value="count">{{ count }} 个</option></select></div><span class="character-count">{{ question.length }}/2000</span><button class="send-button" type="submit" :disabled="busy || uploading || documentsLoading || !question.trim()">{{ busy ? '生成中…' : '发送问题' }}</button></div>
        </form>
        <p class="composer-note">Enter 发送 · Shift + Enter 换行。回答请结合参考资料核对。</p>
      </section>
    </main>

    <aside class="sources-panel" aria-label="参考资料"><div class="sources-heading"><span class="eyebrow green">EVIDENCE</span><h2>参考资料</h2><p>从原文中，找到回答的依据。</p></div>
      <div v-if="!activeAnswer?.matches.length" class="sources-empty"><span class="empty-lines"><i></i><i></i><i></i></span><strong>等待你的第一个问题</strong><p>回答生成后，相关片段会显示在这里。点击来源编号，可以对照原文。</p></div>
      <div v-else class="source-list"><div class="source-count">当前回答 · {{ activeAnswer.matches.length }} 个检索片段</div><article v-for="match in activeAnswer.matches" :key="sourceKey(match)" class="source-card" :class="{ selected: selectedSource === sourceKey(match) }"><div class="source-card-header"><span>{{ referenceFor(match) ? `[${referenceFor(match).id}] 参考资料` : '检索片段' }}</span><span class="score">{{ Number(match.score).toFixed(4) }}</span></div><h3>{{ match.source }}</h3><div class="chunk-label">片段 {{ match.chunk_id }} · 余弦相似度</div><p>{{ match.text }}</p></article><p class="source-help">相似度用于片段排序，不代表答案正确率。</p></div>
    </aside>
  </div>
</template>
