<template>
  <div class="chat-container">
    <!-- 侧边栏 -->
    <div class="sidebar" :class="{ 'collapsed': sidebarCollapsed }">
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">🤖</span>
          <span class="logo-text" v-show="!sidebarCollapsed">DeepSeek Chat</span>
        </div>
        <button class="collapse-btn" @click="toggleSidebar">
          {{ sidebarCollapsed ? '☰' : '✕' }}
        </button>
      </div>

      <div class="new-chat-section">
        <button class="new-chat-btn" @click="startNewChat">
          <span class="icon">+</span>
          <span v-show="!sidebarCollapsed">新对话</span>
        </button>
      </div>

      <div class="chat-history">
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
        <button class="clear-history-btn" @click="clearAllHistory">
          <span class="icon">🗑</span>
          <span v-show="!sidebarCollapsed">清空对话</span>
        </button>
      </div>
    </div>

    <!-- 主聊天区域 -->
    <div class="main-content">
      <!-- 顶部栏 -->
      <div class="top-bar">
        <div class="model-selector">
          <select v-model="selectedModel" class="model-select">
            <option value="deepseek-chat">DeepSeek Chat</option>
            <option value="deepseek-coder">DeepSeek Coder</option>
          </select>
        </div>
      </div>

      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome-message">
          <div class="welcome-icon">🤖</div>
          <h1>你好，我是 DeepSeek</h1>
          <p>有什么可以帮你的吗？</p>
        </div>

        <div v-for="(message, index) in messages" :key="index" class="message" :class="message.role">
          <div class="message-avatar">
            {{ message.role === 'user' ? '👤' : '🤖' }}
          </div>
          <div class="message-content">
            <div class="message-text" v-html="renderMessage(message.content)"></div>
            <div class="message-meta">
              <span class="message-time">{{ formatTime(message.timestamp) }}</span>
            </div>
          </div>
        </div>

        <!-- 加载中提示 -->
        <div v-if="isTyping" class="message assistant">
          <div class="message-avatar">🤖</div>
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
import { sendChatMessage } from '@/api/chat'

// 响应式数据
const messages = ref([])
const userInput = ref('')
const isTyping = ref(false)
const sidebarCollapsed = ref(false)
const selectedModel = ref('deepseek-chat')
const messagesContainer = ref(null)
const inputRef = ref(null)

// 聊天历史
const chatHistory = ref([])
const currentChatId = ref(null)

// 切换侧边栏
const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

// 渲染消息（支持 Markdown）
const renderMessage = (content) => {
  const html = marked.parse(content)
  return DOMPurify.sanitize(html)
}

// 格式化时间
const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  return date.toLocaleDateString()
}

// 处理 Enter 键
const handleEnter = (e) => {
  if (e.shiftKey) {
    return // Shift+Enter 换行
  }
  sendMessage()
}

// 发送消息
const sendMessage = async () => {
  if (!userInput.value.trim() || isTyping.value) return

  const userMessage = {
    role: 'user',
    content: userInput.value.trim(),
    timestamp: Date.now()
  }

  messages.value.push(userMessage)
  userInput.value = ''
  isTyping.value = true

  await scrollToBottom()

  try {
    // 调用 API 发送消息
    const response = await sendChatMessage(userMessage.content, selectedModel.value)

    const assistantMessage = {
      role: 'assistant',
      content: response,
      timestamp: Date.now()
    }

    messages.value.push(assistantMessage)
  } catch (error) {
    console.error('发送消息失败:', error)
    const errorMessage = {
      role: 'assistant',
      content: '抱歉，发生错误：' + (error.message || '未知错误'),
      timestamp: Date.now()
    }
    messages.value.push(errorMessage)
  } finally {
    isTyping.value = false
    await scrollToBottom()

    // 保存聊天历史
    saveChatHistory()
  }
}

// 滚动到底部
const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// 开始新对话
const startNewChat = () => {
  if (messages.value.length > 0) {
    // 保存当前对话
    const title = messages.value[0].content.substring(0, 20) + '...'
    chatHistory.value.unshift({
      id: Date.now(),
      title,
      messages: [...messages.value]
    })
  }
  messages.value = []
  currentChatId.value = null
}

// 切换对话
const switchChat = (chatId) => {
  const chat = chatHistory.value.find(c => c.id === chatId)
  if (chat) {
    currentChatId.value = chatId
    messages.value = [...chat.messages]
  }
}

// 删除对话
const deleteChat = (index) => {
  chatHistory.value.splice(index, 1)
}

// 清空所有历史
const clearAllHistory = () => {
  if (confirm('确定要清空所有对话吗？')) {
    chatHistory.value = []
    messages.value = []
    currentChatId.value = null
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
})

// 配置 marked
marked.setOptions({
  breaks: true,
  gfm: true
})
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background-color: #f5f5f5;
}

/* 侧边栏 */
.sidebar {
  width: 260px;
  background-color: #2d2d2d;
  color: #ffffff;
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar-header {
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #404040;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: bold;
}

.logo-icon {
  font-size: 24px;
}

.collapse-btn {
  background: none;
  border: none;
  color: #ffffff;
  font-size: 20px;
  cursor: pointer;
  padding: 5px;
}

.new-chat-section {
  padding: 15px;
}

.new-chat-btn {
  width: 100%;
  padding: 12px;
  background-color: #4a4a4a;
  color: #ffffff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  transition: background-color 0.2s;
}

.new-chat-btn:hover {
  background-color: #5a5a5a;
}

.new-chat-btn .icon {
  font-size: 18px;
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.history-item {
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 5px;
  transition: background-color 0.2s;
}

.history-item:hover {
  background-color: #404040;
}

.history-item.active {
  background-color: #3a3a3a;
  border-left: 3px solid #4a9eff;
}

.chat-title {
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.delete-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 5px;
  font-size: 16px;
}

.delete-btn:hover {
  color: #ff6b6b;
}

.sidebar-footer {
  padding: 15px;
  border-top: 1px solid #404040;
}

.clear-history-btn {
  width: 100%;
  padding: 12px;
  background-color: #4a4a4a;
  color: #ffffff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}

.clear-history-btn:hover {
  background-color: #5a5a5a;
}

/* 主内容区 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
}

.top-bar {
  padding: 15px 20px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: flex-end;
}

.model-select {
  padding: 8px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  background-color: #ffffff;
  font-size: 14px;
  cursor: pointer;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.welcome-message {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: #666;
}

.welcome-icon {
  font-size: 64px;
  margin-bottom: 20px;
}

.welcome-message h1 {
  font-size: 28px;
  margin-bottom: 10px;
  color: #333;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: #f0f0f0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}

.message.assistant .message-avatar {
  background-color: #4a9eff;
}

.message-content {
  max-width: 70%;
  background-color: #f5f5f5;
  padding: 12px 16px;
  border-radius: 12px;
}

.message.user .message-content {
  background-color: #4a9eff;
  color: #ffffff;
}

.message-text {
  line-height: 1.6;
  word-wrap: break-word;
}

.message-text :deep(pre) {
  background-color: #f0f0f0;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 8px 0;
}

.message-text :deep(code) {
  background-color: #f0f0f0;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
}

.message-text :deep(p) {
  margin: 8px 0;
}

.message-text :deep(ul),
.message-text :deep(ol) {
  padding-left: 20px;
  margin: 8px 0;
}

.message-text :deep(li) {
  margin: 4px 0;
}

.message-text :deep(blockquote) {
  border-left: 4px solid #4a9eff;
  padding-left: 12px;
  margin: 8px 0;
  color: #666;
}

.message-meta {
  display: flex;
  align-items: center;
  margin-top: 8px;
  font-size: 12px;
  color: #999;
}

.message.user .message-meta {
  justify-content: flex-end;
}

/* 输入指示器 */
.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 10px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background-color: #999;
  border-radius: 50%;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-10px);
  }
}

/* 输入区域 */
.input-container {
  border-top: 1px solid #e0e0e0;
  padding: 20px;
}

.input-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.message-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
  font-family: inherit;
  resize: none;
  outline: none;
  transition: border-color 0.2s;
  max-height: 200px;
}

.message-input:focus {
  border-color: #4a9eff;
}

.send-btn {
  width: 48px;
  height: 48px;
  background-color: #4a9eff;
  color: #ffffff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  transition: background-color 0.2s;
}

.send-btn:hover {
  background-color: #3a8eef;
}

.send-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.input-footer {
  margin-top: 8px;
  text-align: center;
}

.hint {
  font-size: 12px;
  color: #999;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    position: absolute;
    left: -260px;
    z-index: 1000;
  }

  .sidebar.collapsed {
    left: 0;
  }

  .message-content {
    max-width: 85%;
  }
}
</style>
