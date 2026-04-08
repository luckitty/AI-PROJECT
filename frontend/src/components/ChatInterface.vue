<template>
  <div class="chat-container">
    <!-- 侧边栏 -->
    <div class="sidebar" :class="{ 'collapsed': sidebarCollapsed }">
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">🤖</span>
          <span class="logo-text" v-show="!sidebarCollapsed">AI Chat</span>
        </div>
        <button
          type="button"
          class="collapse-btn"
          :title="sidebarCollapsed ? '展开侧栏' : '收起侧栏'"
          :aria-expanded="!sidebarCollapsed"
          @click="toggleSidebar"
        >
          <span class="collapse-icon" aria-hidden="true">{{ sidebarCollapsed ? '›' : '‹' }}</span>
        </button>
      </div>

      <div class="new-chat-section">
        <button
          type="button"
          class="new-chat-btn"
          :title="sidebarCollapsed ? '新对话' : undefined"
          @click="startNewChat"
        >
          <span class="icon">+</span>
          <span v-show="!sidebarCollapsed">新对话</span>
        </button>
      </div>

      <div class="chat-history" :class="{ 'is-collapsed': sidebarCollapsed }">
        <div v-if="!sidebarCollapsed">
          <div
            v-for="(chat, index) in chatHistory"
            :key="index"
            class="history-item"
            :class="{ 'active': currentChatId === chat.id }"
            @click="switchChat(chat.id)"
          >
            <span class="chat-title">{{ chat.title }}</span>
            <button class="delete-btn" @click.stop="deleteChat(index)">✕</button>
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <button
          type="button"
          class="clear-history-btn"
          :title="sidebarCollapsed ? '清空对话' : undefined"
          @click="clearAllHistory"
        >
          <span class="icon">🗑</span>
          <span v-show="!sidebarCollapsed">清空对话</span>
        </button>
      </div>
    </div>

    <!-- 主聊天区域 -->
    <div class="main-content">
      <!-- 顶部栏 -->
      <div class="top-bar">
        <div class="reply-mode-selector">
          <label for="reply-mode">回复方式</label>
          <select id="reply-mode" v-model="replyMode" class="model-select">
            <option value="stream">流式</option>
            <option value="plain">一次性</option>
          </select>
        </div>
      </div>

      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome-message">
          <h1>你好，我是 AI 助手</h1>
          <p>有什么可以帮你的吗？</p>
        </div>

        <div v-for="(message, index) in messages" :key="index" class="message" :class="message.role">
          <div v-if="message.role === 'assistant'" class="message-avatar" aria-hidden="true">🤖</div>
          <div class="message-content">
            <div
              v-if="message.role === 'assistant' && message.streaming"
              class="message-text message-text-streaming"
            >
              <span class="stream-plain">{{ message.content }}</span><span class="stream-cursor">▍</span>
            </div>
            <div v-else class="message-text" v-html="renderMessage(message.content || '')"></div>
          </div>
        </div>

        <!-- 加载中提示 三个点 -->
        <div v-if="isTyping" class="message assistant">
          <div class="message-avatar" aria-hidden="true">🤖</div>
          <div class="message-content">
            <div class="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="input-container">
        <div class="input-wrapper">
          <textarea
            v-model="userInput"
            @keydown.enter.prevent="handleEnter"
            @input="adjustInputHeight"
            placeholder="输入你的问题..."
            class="message-input"
            rows="1"
            ref="inputRef"
          ></textarea>
          <button
            class="send-btn"
            @click="sendMessage"
            :disabled="!userInput.trim() || isTyping"
          >
            <span>→</span>
          </button>
        </div>
        <div class="input-footer">
          <span class="hint">Enter 发送，Shift+Enter 换行</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { sendChatMessage, sendChatMessageStream } from '@/api/chat'

// 响应式数据
const messages = ref([])
const userInput = ref('')
const isTyping = ref(false)
const sidebarCollapsed = ref(false)
/** 与后端约定，顶栏已去掉模型切换时仍传默认模型 */
const DEFAULT_CHAT_MODEL = 'deepseek-chat'
const replyMode = ref('stream')
const messagesContainer = ref(null)
const inputRef = ref(null)

// 聊天历史
const chatHistory = ref([])
const currentChatId = ref(null)
// 与后端 LangGraph thread_id 一致，用于多轮对话
const sessionId = ref(null)
const userId = ref(null)
// 长期记忆依赖稳定 user_id；持久化到 localStorage，避免刷新后变成新用户
const storedUserId = localStorage.getItem('chatUserId')
if (storedUserId) {
  userId.value = storedUserId
} else {
  userId.value = crypto.randomUUID()
  localStorage.setItem('chatUserId', userId.value)
}

// 切换侧边栏
const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

// 流式进行中用纯文本分支，不走 marked（见模板 message.streaming）
const renderMessage = (content) => {
  if (!content) return ''
  const html = marked.parse(content)
  return DOMPurify.sanitize(html)
}

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const INPUT_MAX_HEIGHT = 200
const INPUT_MIN_HEIGHT = 46

const adjustInputHeight = () => {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  const next = Math.min(Math.max(el.scrollHeight, INPUT_MIN_HEIGHT), INPUT_MAX_HEIGHT)
  el.style.height = `${next}px`
}

// 处理 Enter 键
const handleEnter = (e) => {
  if (e.shiftKey) {
    return // Shift+Enter 换行
  }
  sendMessage()
}

/** 提交用户句、三个点、session；失败返回 null */
async function startUserTurn() {
  if (!userInput.value.trim() || isTyping.value) return null
  const userMessage = {
    role: 'user',
    content: userInput.value.trim(),
    timestamp: Date.now()
  }
  messages.value.push(userMessage)
  userInput.value = ''
  isTyping.value = true
  if (!sessionId.value) sessionId.value = crypto.randomUUID()
  await nextTick()
  adjustInputHeight()
  await scrollToBottom()
  return userMessage
}

async function endUserTurn() {
  isTyping.value = false
  await scrollToBottom()
  saveChatHistory()
}

function pushErrorAssistant(msg) {
  messages.value.push({
    role: 'assistant',
    content: msg,
    timestamp: Date.now()
  })
}

function formatSendError(e) {
  return '抱歉，发生错误：' + (e.message || '未知错误')
}

/**
 * 流式回复的「打字机」：SSE 可能一次推一大段，先放进 pending，再按帧写入气泡。
 * 返回 { push, flushRest, finalize, abortForError }，供 sendMessageStream 调用。
 */
function createStreamTypewriter() {
  let aiIndex = -1
  let pending = ''
  let rafId = null
  const PUMP = 3 // 每帧显示字数（越小越慢）
  const DRAIN = 12 // 流结束后加快排空 pending

  const stopRaf = () => {
    if (rafId != null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
  }

  const pump = () => {
    rafId = null
    if (aiIndex < 0 || pending.length === 0) return
    const row = messages.value[aiIndex]
    const n = Math.min(PUMP, pending.length)
    row.content += pending.slice(0, n)
    pending = pending.slice(n)
    void nextTick(() => scrollToBottom())
    if (pending.length > 0) rafId = requestAnimationFrame(pump)
  }

  const ensureRow = () => {
    if (aiIndex >= 0) return
    isTyping.value = false
    messages.value.push({
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      streaming: true
    })
    aiIndex = messages.value.length - 1
  }

  return {
    push(text) {
      ensureRow()
      pending += text
      if (rafId == null) rafId = requestAnimationFrame(pump)
    },
    /** 流结束：停掉 pump，把队列里剩余字符尽快画完 */
    async flushRest() {
      stopRaf()
      if (aiIndex < 0) return
      await new Promise((resolve) => {
        const tick = () => {
          if (pending.length === 0) {
            resolve()
            return
          }
          const row = messages.value[aiIndex]
          const n = Math.min(DRAIN, pending.length)
          row.content += pending.slice(0, n)
          pending = pending.slice(n)
          void nextTick(() => scrollToBottom())
          requestAnimationFrame(tick)
        }
        tick()
      })
    },
    /** 出错：停 RAF，把 pending 一次性拼上，避免半截丢字 */
    abortForError() {
      stopRaf()
      if (aiIndex >= 0 && pending) {
        const row = messages.value[aiIndex]
        if (row) row.content += pending
        pending = ''
      }
      isTyping.value = false
      if (aiIndex >= 0) {
        const row = messages.value[aiIndex]
        row.streaming = false
        if (!String(row.content || '').trim()) messages.value.splice(aiIndex, 1)
      }
    },
    /** 正常收尾：关 streaming，若界面仍空则用完整串兜底 */
    finalize(full) {
      if (aiIndex < 0) {
        isTyping.value = false
        messages.value.push({
          role: 'assistant',
          content: full || '',
          timestamp: Date.now()
        })
        return
      }
      const row = messages.value[aiIndex]
      if (!String(row.content || '').trim() && full) row.content = full
      row.streaming = false
    }
  }
}

/** 一次性：await sendChatMessage，整段回复；等待时三个点 */
const sendMessagePlain = async () => {
  const userMessage = await startUserTurn()
  if (!userMessage) return
  try {
    const reply = await sendChatMessage(userMessage.content, DEFAULT_CHAT_MODEL, sessionId.value, userId.value)
    isTyping.value = false
    messages.value.push({
      role: 'assistant',
      content: reply,
      timestamp: Date.now()
    })
  } catch (e) {
    console.error(e)
    isTyping.value = false
    pushErrorAssistant(formatSendError(e))
  } finally {
    await endUserTurn()
  }
}

/** 流式：sendChatMessageStream + 打字机；首包到之前三个点 */
const sendMessageStream = async () => {
  console.log("sendMessageStream===========流式发送消息 \n", userId.value, "\n\n")
  const userMessage = await startUserTurn()
  if (!userMessage) return
  const tw = createStreamTypewriter()
  try {
    const full = await sendChatMessageStream(
      userMessage.content,
      DEFAULT_CHAT_MODEL,
      sessionId.value,
      userId.value,
      tw.push
    )
    await tw.flushRest()
    tw.finalize(full)
  } catch (e) {
    console.error(e)
    tw.abortForError()
    pushErrorAssistant(formatSendError(e))
  } finally {
    await endUserTurn()
  }
}

const sendMessage = async () => {
  if (replyMode.value === 'plain') await sendMessagePlain()
  else await sendMessageStream()
}

// 开始新对话
const startNewChat = () => {
  if (messages.value.length > 0) {
    const firstText = messages.value[0].content
    const title = firstText.length > 20 ? `${firstText.substring(0, 20)}...` : firstText
    const existing =
      currentChatId.value != null
        ? chatHistory.value.find((c) => c.id === currentChatId.value)
        : null

    if (existing) {
      // 当前会话已在侧边栏：只更新内容与 session，避免重复插入
      existing.title = title
      existing.messages = [...messages.value]
      if (sessionId.value) {
        existing.sessionId = sessionId.value
      }
    } else if (currentChatId.value == null) {
      // 仅「未归档」的临时会话才新增一条历史
      chatHistory.value.unshift({
        id: Date.now(),
        sessionId: sessionId.value,
        title,
        messages: [...messages.value]
      })
    }
    saveChatHistory()
  }
  messages.value = []
  currentChatId.value = null
  sessionId.value = null
}

// 切换对话
const switchChat = (chatId) => {
  const chat = chatHistory.value.find(c => c.id === chatId)
  if (chat) {
    currentChatId.value = chatId
    messages.value = [...chat.messages]
    sessionId.value = chat.sessionId || crypto.randomUUID()
    if (!chat.sessionId) {
      chat.sessionId = sessionId.value
      saveChatHistory()
    }
  }
}

// 删除对话
const deleteChat = (index) => {
  const removed = chatHistory.value[index]
  chatHistory.value.splice(index, 1)
  if (removed && currentChatId.value === removed.id) {
    messages.value = []
    sessionId.value = null
    currentChatId.value = null
  }
  saveChatHistory()
}

// 清空所有历史
const clearAllHistory = () => {
  if (confirm('确定要清空所有对话吗？')) {
    chatHistory.value = []
    messages.value = []
    currentChatId.value = null
    sessionId.value = null
    saveChatHistory()
  }
}

// 保存聊天历史到本地存储
const saveChatHistory = () => {
  try {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory.value))
  } catch (e) {
    console.error('保存聊天历史失败:', e)
  }
}

// 加载聊天历史
const loadChatHistory = () => {
  try {
    const saved = localStorage.getItem('chatHistory')
    if (saved) {
      chatHistory.value = JSON.parse(saved)
    }
  } catch (e) {
    console.error('加载聊天历史失败:', e)
  }
}

// 组件挂载时加载历史
onMounted(() => {
  loadChatHistory()
  inputRef.value?.focus()
  nextTick(() => adjustInputHeight())
})

// 配置 marked
marked.setOptions({
  breaks: true,
  gfm: true
})
</script>

<style scoped src="../styles/chatInterface.css"></style>
