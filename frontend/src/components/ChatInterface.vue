<template>
  <div class="chat-container">
    <!-- 侧栏：展开 / 收起、新对话、搜索、分组列表、行内删除、底栏用户菜单 -->
    <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-toolbar">
        <!--
          对齐 ChatGPT：展开态为 Logo + 收起按钮；收起态顶栏仅保留 Logo，
          悬停 Logo 区域再显示「展开侧栏」图标（半透明遮罩 + 图标），点击任意时刻均可展开。
        -->
        <div class="sidebar-brand-wrap" :class="{ 'is-collapsed-toolbar': sidebarCollapsed }">
          <img class="sidebar-brand-logo" src="/app-logo.png" alt="旅游攻略助手" />
          <button
            v-if="sidebarCollapsed"
            type="button"
            class="sidebar-expand-from-logo"
            title="展开侧栏"
            aria-label="展开侧栏"
            aria-expanded="false"
            @click="toggleSidebar"
          >
            <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <rect x="4" y="4" width="6" height="16" rx="1.5" stroke-linejoin="round" />
              <path d="M14 6h6M14 12h6M14 18h6" stroke-linecap="round" />
            </svg>
          </button>
        </div>
        <button
          v-show="!sidebarCollapsed"
          type="button"
          class="sidebar-icon-btn sidebar-toggle"
          title="收起侧栏"
          :aria-expanded="true"
          @click="toggleSidebar"
        >
          <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <rect x="4" y="4" width="6" height="16" rx="1.5" stroke-linejoin="round" />
            <path d="M14 6h6M14 12h6M14 18h6" stroke-linecap="round" />
          </svg>
        </button>
      </div>

      <div class="sidebar-new-chat-wrap">
        <button
          type="button"
          class="sidebar-new-chat"
          :title="sidebarCollapsed ? '新对话' : undefined"
          @click="startNewChat"
        >
          <!-- 方底铅笔：贴近 ChatGPT「新聊天」 -->
          <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L8 18l-4 1 1-4L16.5 3.5z" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <span v-show="!sidebarCollapsed" class="sidebar-new-chat-label">新对话</span>
        </button>
      </div>

      <div v-show="!sidebarCollapsed" class="sidebar-search-wrap">
        <label class="sidebar-search-label visually-hidden" for="sidebar-search-input">搜索对话</label>
        <div class="sidebar-search-inner">
          <svg class="sidebar-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-3.5-3.5" stroke-linecap="round" />
          </svg>
          <input
            id="sidebar-search-input"
            v-model="sidebarSearchQuery"
            type="search"
            class="sidebar-search-input"
            placeholder="搜索对话"
            autocomplete="off"
            spellcheck="false"
          />
        </div>
      </div>

      <div class="chat-history" :class="{ 'is-collapsed': sidebarCollapsed }">
        <template v-if="!sidebarCollapsed">
          <template v-if="chatSidebarGroups.length === 0">
            <p class="sidebar-empty-hint">{{ sidebarSearchQuery.trim() ? '没有匹配的对话' : '暂无对话记录' }}</p>
          </template>
          <template v-else>
            <template v-for="group in chatSidebarGroups" :key="group.key">
              <div class="history-group-label">{{ group.label }}</div>
              <div
                v-for="chat in group.items"
                :key="chat.id"
                class="history-row-wrap"
              >
                <div
                  class="history-row"
                  :class="{ active: currentChatId === chat.id }"
                  role="button"
                  tabindex="0"
                  @click="switchChat(chat.id)"
                  @keydown.enter.prevent="switchChat(chat.id)"
                >
                  <span class="history-row-title">{{ chat.title }}</span>
                  <button
                    type="button"
                    class="history-delete-btn"
                    title="删除对话"
                    aria-label="删除对话"
                    @click.stop="deleteChatById(chat.id)"
                  >
                    <svg class="icon-svg-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                      <path d="M18 6L6 18M6 6l12 12" stroke-linecap="round" />
                    </svg>
                  </button>
                </div>
              </div>
            </template>
          </template>
        </template>
      </div>

      <div class="sidebar-footer">
        <button
          type="button"
          class="sidebar-footer-trigger"
          :title="sidebarCollapsed ? accountShortLabel : undefined"
          :aria-expanded="footerMenuOpen"
          aria-haspopup="menu"
          @click.stop="onSidebarFooterClick"
        >
          <span class="sidebar-footer-avatar" aria-hidden="true">{{ accountShortLabel }}</span>
          <span v-show="!sidebarCollapsed" class="sidebar-footer-name">旅游助手</span>
          <svg
            v-show="!sidebarCollapsed"
            class="sidebar-footer-chevron"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            aria-hidden="true"
          >
            <path d="M6 9l6 6 6-6" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <div v-if="footerMenuOpen && !sidebarCollapsed" class="sidebar-footer-menu" role="menu" @click.stop>
          <button type="button" class="history-menu-item" role="menuitem" @click="clearAllHistoryAndCloseMenu">
            清空所有对话
          </button>
        </div>
      </div>
    </aside>

    <!-- 小屏侧栏打开时的遮罩，点击关闭（对齐常见抽屉交互） -->
    <div
      v-if="showSidebarBackdrop"
      class="sidebar-backdrop"
      aria-hidden="true"
      @click="toggleSidebar"
    ></div>

    <!-- 主区：居中对话列 + 底部 Composer -->
    <main class="main-content">
      <button
        v-if="showMobileSidebarTrigger"
        type="button"
        class="mobile-sidebar-trigger"
        aria-label="打开侧栏"
        @click="toggleSidebar"
      >
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M4 6h16M4 12h16M4 18h10" stroke-linecap="round" />
        </svg>
      </button>

      <div class="messages-container" ref="messagesContainer" @click="onMessagesContainerClick">
        <div v-if="messages.length === 0" class="welcome-message">
          <div class="welcome-inner">
            <!-- 品牌 Logo：静态资源置于 frontend/public，构建后由根路径提供 -->
            <img class="welcome-logo" src="/app-logo.png" alt="旅游攻略助手" />
            <h1 class="welcome-title">想去哪里旅游？</h1>
            <p class="welcome-sub">我是旅游攻略助手，行程、路线、景点都可以问我。</p>
          </div>
        </div>

        <div
          v-for="(message, index) in messages"
          :key="index"
          class="message-row"
          :class="message.role"
        >
          <div class="message-inner">
            <!-- 助手回复不展示头像，正文直接与居中列对齐 -->
            <div class="message-content">
              <div
                v-if="message.role === 'assistant' && message.streaming"
                class="message-text message-text-streaming"
              >
                <!-- 流式阶段也实时走 Markdown 渲染，避免“生成中无格式、结束后才有格式”的割裂体验 -->
                <div class="stream-markdown" v-html="message.streamingHtml || ''"></div>
              </div>
              <div v-else class="message-text" v-html="message.html || ''"></div>
            </div>
          </div>
        </div>

        <div v-if="isTyping" class="message-row assistant">
          <div class="message-inner">
            <div class="message-content">
              <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="composer-wrap">
        <div class="composer-inner">
          <div class="input-container">
            <div class="input-shell">
              <textarea
                v-model="userInput"
                @keydown.enter="handleEnter"
                @input="adjustInputHeight"
                placeholder="给助手发消息…"
                class="message-input"
                rows="1"
                ref="inputRef"
              ></textarea>
              <button
                type="button"
                class="send-btn"
                :class="{ stop: isTyping }"
                @click="handleSendOrStop"
                :disabled="!isTyping && !userInput.trim()"
                :title="isTyping ? '停止生成' : '发送'"
              >
                <!-- 纸飞机：示意发送，视觉上比旧三角箭头更清晰 -->
                <svg v-if="!isTyping" class="send-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path
                    d="M3.478 2.405a.75.75 0 0 0-.926.94l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.405Z"
                  />
                </svg>
                <svg v-else class="stop-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
              </button>
            </div>
            <p class="input-hint">内容由 AI 生成，请核对重要信息。Enter 发送 · Shift+Enter 换行</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import {
  abortChatRequest,
  isAbortRequestError,
  sendChatMessageStream,
  stopChatSession,
  stopChatSessionKeepalive
} from '@/api/chat'

// 响应式数据
const messages = ref([])
const userInput = ref('')
const isTyping = ref(false)
const sidebarCollapsed = ref(false)
/** 用于移动端布局：≤768px 视为小屏，侧栏改为抽屉并默认收起 */
const MOBILE_BREAKPOINT_PX = 768
const windowWidth = ref(typeof window !== 'undefined' ? window.innerWidth : MOBILE_BREAKPOINT_PX + 1)

const isMobileLayout = computed(() => windowWidth.value <= MOBILE_BREAKPOINT_PX)
/** 小屏且侧栏收起时显示左上角菜单键（侧栏滑出后该按钮隐藏，避免重复） */
const showMobileSidebarTrigger = computed(() => isMobileLayout.value && sidebarCollapsed.value)
/** 小屏且侧栏展开时显示半透明遮罩 */
const showSidebarBackdrop = computed(() => isMobileLayout.value && !sidebarCollapsed.value)

function syncWindowWidth() {
  if (typeof window === 'undefined') return
  windowWidth.value = window.innerWidth
}
/** 与后端约定，顶栏已去掉模型切换时仍传默认模型 */
const DEFAULT_CHAT_MODEL = 'deepseek-chat'
const messagesContainer = ref(null)
const inputRef = ref(null)
const autoScrollEnabled = ref(true)

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

/** 底栏头像两字：从稳定 user_id 派生，贴近 ChatGPT 账户缩写观感 */
const accountShortLabel = computed(() => {
  const raw = String(userId.value || '').replace(/[^a-zA-Z0-9]/g, '')
  if (raw.length >= 2) return raw.slice(0, 2).toUpperCase()
  return 'ME'
})

// 侧栏搜索、底栏菜单
const sidebarSearchQuery = ref('')
const footerMenuOpen = ref(false)

function toggleFooterMenu() {
  footerMenuOpen.value = !footerMenuOpen.value
}

/**
 * 收起态只显示头像：第一次点击先展开侧栏（否则无法点到搜索等功能），展开后再走账户菜单。
 */
function onSidebarFooterClick() {
  if (sidebarCollapsed.value) {
    sidebarCollapsed.value = false
    return
  }
  toggleFooterMenu()
}

function clearAllHistoryAndCloseMenu() {
  footerMenuOpen.value = false
  clearAllHistory()
}

function onSidebarDocPointerDown(e) {
  const el = e.target
  if (!el.closest) return
  if (el.closest('.sidebar-footer-trigger')) return
  if (el.closest('.sidebar-footer-menu')) return
  footerMenuOpen.value = false
}

function onSidebarKeydown(e) {
  if (e.key !== 'Escape') return
  footerMenuOpen.value = false
  if (isMobileLayout.value && !sidebarCollapsed.value) {
    sidebarCollapsed.value = true
  }
}

/** 本地日开始时间戳，用于「今天 / 昨天 / 7 天」分组 */
function startOfLocalDay(ts) {
  const d = new Date(ts)
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

const chatsForSidebar = computed(() => {
  const q = sidebarSearchQuery.value.trim().toLowerCase()
  let list = chatHistory.value.slice()
  if (q) list = list.filter((c) => String(c.title || '').toLowerCase().includes(q))
  list.sort((a, b) => (b.updatedAt ?? b.id ?? 0) - (a.updatedAt ?? a.id ?? 0))
  return list
})

/** 对话列表分组：贴近 ChatGPT 的时间簇展示 */
const chatSidebarGroups = computed(() => {
  const list = chatsForSidebar.value
  const today0 = startOfLocalDay(Date.now())
  const dayMs = 86400000
  const yesterday0 = today0 - dayMs
  const weekStart = today0 - 7 * dayMs
  const today = []
  const yesterday = []
  const week = []
  const older = []
  for (const chat of list) {
    const t = chat.updatedAt ?? chat.id ?? 0
    if (t >= today0) today.push(chat)
    else if (t >= yesterday0) yesterday.push(chat)
    else if (t >= weekStart) week.push(chat)
    else older.push(chat)
  }
  const out = []
  if (today.length) out.push({ key: 'today', label: '今天', items: today })
  if (yesterday.length) out.push({ key: 'yesterday', label: '昨天', items: yesterday })
  if (week.length) out.push({ key: 'week', label: '过去 7 天', items: week })
  if (older.length) out.push({ key: 'older', label: '更早', items: older })
  return out
})

/** 把指定会话置顶并刷新活动时间，模拟 ChatGPT 最近对话上浮 */
function bumpChatToTop(chatId) {
  const i = chatHistory.value.findIndex((c) => c.id === chatId)
  if (i < 0) return
  const row = chatHistory.value[i]
  row.updatedAt = Date.now()
  if (i === 0) return
  chatHistory.value.splice(i, 1)
  chatHistory.value.unshift(row)
}

/** 按会话 id 删除一条本地历史（叉号触发，不经过索引以免搜索折叠后错位） */
function deleteChatById(chatId) {
  const idx = chatHistory.value.findIndex((c) => c.id === chatId)
  if (idx < 0) return
  deleteChat(idx)
}

// 切换侧边栏
const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
  footerMenuOpen.value = false
}

// chat-reply 链接协议：Markdown 链接触发前端代发用户短回复（如攻略末尾「可以」）。
const QUICK_REPLY_LINK_RE = /<a href="chat-reply:([^"]+)">([\s\S]*?)<\/a>/g

function convertQuickReplyLinks(html) {
  return html.replace(QUICK_REPLY_LINK_RE, (_, replyEncoded, label) => {
    const reply = decodeURIComponent(replyEncoded)
    const safeReply = reply.replace(/"/g, '&quot;')
    return `<button type="button" class="chat-quick-reply" data-reply="${safeReply}">${label}</button>`
  })
}

// 统一 Markdown 渲染入口：普通消息使用缓存 html，流式消息使用 streamingHtml。
const renderMessage = (content) => {
  if (!content) return ''
  const html = marked.parse(content)
  const withQuickReplies = convertQuickReplyLinks(html)
  return DOMPurify.sanitize(withQuickReplies, { ADD_ATTR: ['data-reply'] })
}

// 统一创建可渲染消息，提前缓存 html，避免模板阶段反复执行 markdown 解析。
function buildRenderedMessage(role, content, extra = {}) {
  const safeContent = content || ''
  return {
    role,
    content: safeContent,
    html: renderMessage(safeContent),
    timestamp: Date.now(),
    ...extra
  }
}

// 从本地历史恢复时补齐渲染字段，并移除不应保留的流式中间态字段。
function normalizeMessagesForRender(list) {
  return (list || []).map((message) => {
    const safeContent = String(message?.content || '')
    return {
      ...message,
      content: safeContent,
      html: renderMessage(safeContent),
      streaming: false,
      streamingHtml: ''
    }
  })
}

const AUTO_SCROLL_BOTTOM_GAP = 60

function isNearBottom() {
  const el = messagesContainer.value
  if (!el) return true
  const gap = el.scrollHeight - el.scrollTop - el.clientHeight
  return gap <= AUTO_SCROLL_BOTTOM_GAP
}

function updateAutoScrollStateByUserPosition() {
  autoScrollEnabled.value = isNearBottom()
}

const scrollToBottom = async (force = false) => {
  if (!force && !autoScrollEnabled.value) return
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

// 处理 Enter 键：发送前须排除 IME 组字态，否则拼音未确认时按 Enter 会误发
const handleEnter = (e) => {
  if (e.shiftKey) {
    return // Shift+Enter 换行：不 preventDefault，交给浏览器插入换行
  }
  // isComposing：搜狗/系统拼音等组字过程中 Enter 用于确认候选或字母
  // keyCode 229：部分环境下 IME 按键仍可能被当成 Enter，需一并放行
  if (e.isComposing || e.keyCode === 229) {
    return
  }
  e.preventDefault()
  sendMessage()
}

/** 提交用户句、三个点、session；失败返回 null */
async function startUserTurn() {
  if (!userInput.value.trim() || isTyping.value) return null
  const userMessage = buildRenderedMessage('user', userInput.value.trim())
  messages.value.push(userMessage)
  userInput.value = ''
  isTyping.value = true
  autoScrollEnabled.value = true
  if (!sessionId.value) sessionId.value = crypto.randomUUID()
  await nextTick()
  adjustInputHeight()
  await scrollToBottom(true)
  return userMessage
}

async function endUserTurn() {
  isTyping.value = false
  autoScrollEnabled.value = true
  if (currentChatId.value) {
    bumpChatToTop(currentChatId.value)
  }
  await scrollToBottom(true)
  saveChatHistory()
}

function pushErrorAssistant(msg) {
  messages.value.push(buildRenderedMessage('assistant', msg))
}

function formatSendError(e) {
  return '抱歉，发生错误：' + (e.message || '未知错误')
}

/**
 * 中断当前大模型请求，并立即结束前端「正在生成」态。
 * 页面层只负责用户交互状态，网络中断由 api/chat.js 统一处理。
 */
function stopGenerating() {
  // 先通知后端设置 stop 标记，再中断前端流读取，双向确保尽快停。
  if (sessionId.value) {
    stopChatSession(sessionId.value).catch((e) => {
      console.error('通知后端中断失败:', e)
    })
  }
  abortChatRequest()
  isTyping.value = false
}

function stopGeneratingForPageUnload() {
  // 页面刷新/关闭时用 keepalive 兜底通知后端停止当前会话。
  if (sessionId.value) stopChatSessionKeepalive(sessionId.value)
  abortChatRequest()
}

/**
 * 流式回复（对齐常见 ChatGPT 类产品）：SSE 每来一段就立刻拼进气泡，不在前端二次限速。
 * 滚动用 requestAnimationFrame 合并到每帧最多一次，避免极小 chunk 刷屏时反复触发布局。
 * 返回 { push, finalize, abortForError }，供 sendMessageStream 调用。
 */
function createStreamDirectWriter() {
  let aiIndex = -1
  let scrollRafId = null
  let appendRafId = null
  let renderTimerId = null
  let pendingChunkText = ''
  const STREAM_MARKDOWN_RENDER_INTERVAL_MS = 120

  const scheduleScrollToBottom = () => {
    if (scrollRafId != null) return
    scrollRafId = requestAnimationFrame(() => {
      scrollRafId = null
        void nextTick(() => scrollToBottom(false))
    })
  }

  // chunk 到达可能远高于浏览器渲染帧率；按帧合并可减少响应式抖动并让滚动更平滑。
  const scheduleAppendChunksByFrame = () => {
    if (appendRafId != null) return
    appendRafId = requestAnimationFrame(() => {
      appendRafId = null
      if (!pendingChunkText) return
      ensureRow()
      const row = messages.value[aiIndex]
      row.content += pendingChunkText
      pendingChunkText = ''
      scheduleStreamingMarkdownRender()
      scheduleScrollToBottom()
    })
  }

  const ensureRow = () => {
    if (aiIndex >= 0) return
    isTyping.value = false
    messages.value.push({
      ...buildRenderedMessage('assistant', ''),
      streamingHtml: '',
      streaming: true
    })
    aiIndex = messages.value.length - 1
  }

  // 流式内容到达频率可能非常高，Markdown 全量解析改为小间隔节流，降低主线程压力。
  const scheduleStreamingMarkdownRender = () => {
    if (renderTimerId != null) return
    renderTimerId = window.setTimeout(() => {
      renderTimerId = null
      if (aiIndex < 0) return
      const row = messages.value[aiIndex]
      row.streamingHtml = renderMessage(row.content || '')
    }, STREAM_MARKDOWN_RENDER_INTERVAL_MS)
  }

  return {
    /** 网络层解析出的正文增量，直接追加 */
    push(text) {
      if (!text) return
      pendingChunkText += text
      scheduleAppendChunksByFrame()
    },
    /** 出错：关 streaming，无正文则移除空气泡 */
    abortForError() {
      isTyping.value = false
      pendingChunkText = ''
      if (appendRafId != null) {
        cancelAnimationFrame(appendRafId)
        appendRafId = null
      }
      if (renderTimerId != null) {
        clearTimeout(renderTimerId)
        renderTimerId = null
      }
      if (scrollRafId != null) {
        cancelAnimationFrame(scrollRafId)
        scrollRafId = null
      }
      if (aiIndex >= 0) {
        const row = messages.value[aiIndex]
        row.streaming = false
        if (!String(row.content || '').trim()) messages.value.splice(aiIndex, 1)
      }
    },
    /** 正常收尾：关 streaming，若界面仍空则用完整串兜底 */
    finalize(full) {
      if (appendRafId != null) {
        cancelAnimationFrame(appendRafId)
        appendRafId = null
      }
      if (pendingChunkText) {
        ensureRow()
        const row = messages.value[aiIndex]
        row.content += pendingChunkText
        pendingChunkText = ''
      }
      if (renderTimerId != null) {
        clearTimeout(renderTimerId)
        renderTimerId = null
      }
      if (aiIndex < 0) {
        isTyping.value = false
        messages.value.push(buildRenderedMessage('assistant', full || ''))
        return
      }
      const row = messages.value[aiIndex]
      if (!String(row.content || '').trim() && full) row.content = full
      row.streamingHtml = renderMessage(row.content || '')
      row.html = row.streamingHtml
      row.streaming = false
    }
  }
}

/** 流式：sendChatMessageStream + 收到即追加；首包到之前三个点 */
const sendMessageStream = async () => {
  console.log("sendMessageStream===========流式发送消息 \n", userId.value, "\n\n")
  const userMessage = await startUserTurn()
  if (!userMessage) return
  const writer = createStreamDirectWriter()
  try {
    const full = await sendChatMessageStream(
      userMessage.content,
      DEFAULT_CHAT_MODEL,
      sessionId.value,
      userId.value,
      writer.push,
    )
    writer.finalize(full)
  } catch (e) {
    console.error(e)
    if (isAbortRequestError(e)) return
    writer.abortForError()
    pushErrorAssistant(formatSendError(e))
  } finally {
    await endUserTurn()
  }
}

const sendMessage = sendMessageStream

/** 点击消息内快捷回复按钮：填入文案并走正常发送流程。 */
async function sendQuickReply(reply, btn) {
  if (isTyping.value || !String(reply || '').trim()) return
  if (btn) btn.disabled = true
  userInput.value = String(reply).trim()
  await sendMessage()
}

/** 消息区事件委托：捕获 chat-quick-reply 按钮点击。 */
function onMessagesContainerClick(e) {
  const btn = e.target?.closest?.('.chat-quick-reply')
  if (!btn) return
  e.preventDefault()
  void sendQuickReply(btn.getAttribute('data-reply'), btn)
}

/**
 * 发送与停止共用按钮：
 * - 空闲时：发送消息
 * - 生成中：中断请求
 */
const handleSendOrStop = async () => {
  if (isTyping.value) {
    stopGenerating()
    return
  }
  await sendMessage()
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
      existing.messages = normalizeMessagesForRender(messages.value)
      if (sessionId.value) {
        existing.sessionId = sessionId.value
      }
      bumpChatToTop(existing.id)
    } else if (currentChatId.value == null) {
      // 仅「未归档」的临时会话才新增一条历史
      const nid = Date.now()
      chatHistory.value.unshift({
        id: nid,
        updatedAt: nid,
        sessionId: sessionId.value,
        title,
        messages: normalizeMessagesForRender(messages.value)
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
    messages.value = normalizeMessagesForRender(chat.messages)
    sessionId.value = chat.sessionId || crypto.randomUUID()
    if (!chat.sessionId) {
      chat.sessionId = sessionId.value
    }
    // 点击侧栏只打开会话，不置顶、不改 updatedAt（置顶留给真实发消息等活跃行为）
    saveChatHistory()
    footerMenuOpen.value = false
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
    // 本地只存业务字段，剔除 html 等渲染缓存，减少存储体积并避免脏状态持久化。
    const persistedHistory = chatHistory.value.map((chat) => ({
      ...chat,
      messages: (chat.messages || []).map((message) => ({
        role: message.role,
        content: message.content,
        timestamp: message.timestamp
      }))
    }))
    localStorage.setItem('chatHistory', JSON.stringify(persistedHistory))
  } catch (e) {
    console.error('保存聊天历史失败:', e)
  }
}

// 加载聊天历史
const loadChatHistory = () => {
  try {
    const saved = localStorage.getItem('chatHistory')
    if (saved) {
      const parsedHistory = JSON.parse(saved)
      chatHistory.value = (parsedHistory || []).map((chat) => ({
        ...chat,
        updatedAt: chat.updatedAt ?? chat.id ?? Date.now(),
        messages: normalizeMessagesForRender(chat.messages)
      }))
    }
  } catch (e) {
    console.error('加载聊天历史失败:', e)
  }
}

// 组件挂载时加载历史
onMounted(() => {
  syncWindowWidth()
  // 手机首屏默认收起侧栏，避免挡住对话区；桌面保持展开（collapsed 默认 false）
  if (windowWidth.value <= MOBILE_BREAKPOINT_PX) {
    sidebarCollapsed.value = true
  }
  loadChatHistory()
  inputRef.value?.focus()
  nextTick(() => adjustInputHeight())
  messagesContainer.value?.addEventListener('scroll', updateAutoScrollStateByUserPosition, {
    passive: true
  })
  window.addEventListener('resize', syncWindowWidth)
  document.addEventListener('pointerdown', onSidebarDocPointerDown)
  window.addEventListener('keydown', onSidebarKeydown)
  // 刷新、关闭页签或浏览器回收页面时都触发，尽量让后端立即停止旧请求。
  window.addEventListener('beforeunload', stopGeneratingForPageUnload)
  window.addEventListener('pagehide', stopGeneratingForPageUnload)
})

onUnmounted(() => {
  messagesContainer.value?.removeEventListener('scroll', updateAutoScrollStateByUserPosition)
  window.removeEventListener('resize', syncWindowWidth)
  document.removeEventListener('pointerdown', onSidebarDocPointerDown)
  window.removeEventListener('keydown', onSidebarKeydown)
  window.removeEventListener('beforeunload', stopGeneratingForPageUnload)
  window.removeEventListener('pagehide', stopGeneratingForPageUnload)
})

// 配置 marked
marked.setOptions({
  breaks: true,
  gfm: true
})
</script>

<style scoped lang="less" src="../styles/chatInterface.less"></style>
