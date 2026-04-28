import axios from 'axios'

// 配置 API 基础路径
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 当前会话中的大模型请求控制器（同一时刻只允许一个进行中请求）
let activeChatAbortController = null

/**
 * 创建新的中断控制器，并替换旧的进行中请求控制器。
 * 这样页面点击「停止生成」时可以统一中断当前大模型请求（plain/stream 共用）。
 */
function createChatAbortController() {
  activeChatAbortController = new AbortController()
  return activeChatAbortController
}

/**
 * 主动中断当前大模型请求。
 * 若当前没有进行中的请求，则静默返回。
 */
export function abortChatRequest() {
  if (activeChatAbortController) {
    activeChatAbortController.abort()
    activeChatAbortController = null
  }
}

/**
 * 通知后端中断指定会话。
 * 仅中断服务端当前会话，不影响本地历史。
 */
export async function stopChatSession(sessionId) {
  if (!sessionId) return
  await apiClient.get('/api/chat/stop', {
    params: { session_id: sessionId }
  })
}

/**
 * 页面卸载阶段使用 keepalive 异步通知后端中断会话。
 * 刷新/关闭标签页时无法可靠等待 Promise，这里走 fire-and-forget。
 */
export function stopChatSessionKeepalive(sessionId) {
  if (!sessionId) return
  const url = `${API_BASE_URL}/api/chat/stop?session_id=${encodeURIComponent(sessionId)}`
  fetch(url, { method: 'GET', keepalive: true }).catch(() => {})
}

/**
 * 判断错误是否由前端主动中断触发。
 * fetch 会抛 AbortError；axios 在 signal 中断时 code 常为 ERR_CANCELED。
 */
export function isAbortRequestError(error) {
  return error?.name === 'AbortError' || error?.code === 'ERR_CANCELED'
}

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证 token
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API 错误:', error)
    const errorMessage = error.response?.data?.detail ||
                        error.response?.data?.message ||
                        error.message ||
                        '请求失败，请稍后重试'
    return Promise.reject(new Error(errorMessage))
  }
)

/**
 * 发送聊天消息（非流式，POST /api/chat，一次返回完整 reply）。
 * 页面上的「三个点」加载态由 ChatInterface.vue 在 await 本函数前后控制 isTyping。
 */
export const sendChatMessage = async (message, model = 'deepseek-chat', sessionId, userId) => {
  const controller = createChatAbortController()
  try {
    const response = await apiClient.post('/api/chat', {
      message,
      model,
      session_id: sessionId,
      user_id: userId || undefined
    }, {
      signal: controller.signal
    })
    return response.reply || response.content || response.message || '未收到有效回复'
  } catch (error) {
    throw error
  } finally {
    if (activeChatAbortController === controller) activeChatAbortController = null
  }
}

/**
 * 把 SSE 里「data:」后面那一截字符串解析成要给用户看的正文。
 *
 * 后端约定两种形式：
 * 1) JSON：{"content":"..."} 或带 error 字段 —— 一行一条，正文里即使有换行也不会弄坏协议
 * 2) 纯文本：兼容旧格式或非 JSON 行
 *
 * @param {string} raw - 去掉 "data: " 前缀后的内容（可能仍含首尾空白）
 * @returns {string} 拼进完整回复里的一段文字；空串表示本行无正文（可跳过）
 */
function parseSseDataPayload(raw) {
  let parsed = null
  try {
    parsed = JSON.parse(raw)
  } catch {
    parsed = null
  }
  if (parsed && typeof parsed === 'object') {
    // 后端用 JSON 传错误时在这里抛出，外层 catch 会交给页面显示
    if (parsed.error) throw new Error(parsed.error)
    return parsed.content || parsed.message || parsed.reply || ''
  }
  return raw
}

/**
 * 解析单行 SSE 事件，兼容 "data: xxx" 与 "data:xxx" 两种写法。
 * 返回 null 表示该行不是 data 事件或无正文。
 */
function parseSseDataLine(line) {
  const normalized = String(line || '').trim()
  if (!normalized.startsWith('data:')) return null
  const raw = normalized.slice(5).trimStart().trimEnd()
  if (!raw || raw === '[DONE]') return ''
  return parseSseDataPayload(raw)
}

/**
 * 流式请求 POST /api/chat/stream（SSE）。
 * 首包到达前的「三个点」由 ChatInterface.vue 在首次 onChunk 之前保持 isTyping；
 * 每解析出一小段正文会调 onChunk，由页面直接拼进助手气泡（与常见 ChatGPT 类流式一致）。
 */
export const sendChatMessageStream = async (
  message,
  model = 'deepseek-chat',
  sessionId,
  userId,
  onChunk,
) => {
  const controller = createChatAbortController()
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        model,
        session_id: sessionId || undefined,
        user_id: userId || undefined
      }),
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    // fetch 的 body 是 ReadableStream，只能一块块 read，不能保证每次刚好是一行
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let fullResponse = ''
    // 未凑满一行的尾巴先放在 buffer 里，下次 read 再拼上，避免「半行」被当成整行解析错
    let sseBuffer = ''

    while (true) {
      const { done, value } = await reader.read()
      // stream:!done 表示后面还有字节，避免多字节 UTF-8 被截断
      sseBuffer += decoder.decode(value || new Uint8Array(), { stream: !done })
      const lines = sseBuffer.split('\n')
      // 最后一项可能是半行，留到下一轮
      sseBuffer = lines.pop() ?? ''
      for (const line of lines) {
        const piece = parseSseDataLine(line)
        if (piece) {
          fullResponse += piece
          // 交给页面直接追加到助手气泡（流式展示）
          if (typeof onChunk === 'function') onChunk(piece)
        }
      }
      // 结束时把缓冲区最后一行也尝试解析，避免无换行尾包被漏掉或延后到下一次请求。
      if (done && sseBuffer) {
        const tailPiece = parseSseDataLine(sseBuffer)
        if (tailPiece) {
          fullResponse += tailPiece
          if (typeof onChunk === 'function') onChunk(tailPiece)
        }
      }
      if (done) break
    }

    return fullResponse
  } catch (error) {
    console.error('流式请求错误:', error)
    throw error
  } finally {
    if (activeChatAbortController === controller) activeChatAbortController = null
  }
}

/**
 * 获取对话历史
 * @param {string} sessionId - 会话 ID
 * @returns {Promise<Array>} - 对话历史
 */
export const getChatHistory = async (sessionId) => {
  try {
    const response = await apiClient.get(`/api/chat/history/${sessionId}`)
    return response.messages || []
  } catch (error) {
    throw error
  }
}

/**
 * 清空对话历史
 * @param {string} sessionId - 会话 ID
 * @returns {Promise<void>}
 */
export const clearChatHistory = async (sessionId) => {
  try {
    await apiClient.delete(`/api/chat/history/${sessionId}`)
  } catch (error) {
    throw error
  }
}

export default apiClient
