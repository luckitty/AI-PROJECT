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
 * 发送聊天消息
 * @param {string} message - 用户消息
 * @param {string} model - 模型名称
 * @returns {Promise<string>} - AI 回复
 */
export const sendChatMessage = async (message, model = 'deepseek-chat') => {
  try {
    const response = await apiClient.post('/api/chat', {
      message,
      model
    })
    return response.reply || response.content || response.message || '未收到有效回复'
  } catch (error) {
    throw error
  }
}

/**
 * 流式发送聊天消息
 * @param {string} message - 用户消息
 * @param {string} model - 模型名称
 * @param {Function} onChunk - 处理每个数据块的回调
 * @returns {Promise<string>} - 完整的 AI 回复
 */
export const sendChatMessageStream = async (message, model = 'deepseek-chat', onChunk) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message, model })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullResponse = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') break

          try {
            const parsed = JSON.parse(data)
            const content = parsed.content || parsed.message || parsed.reply
            if (content) {
              fullResponse += content
              if (onChunk) {
                onChunk(content)
              }
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }

    return fullResponse
  } catch (error) {
    console.error('流式请求错误:', error)
    throw error
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
