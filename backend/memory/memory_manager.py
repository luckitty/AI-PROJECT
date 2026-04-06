# from memory.long_term_memory import LongTermMemory


# class MemoryManager:

#     def __init__(self):
#         self.long_term = LongTermMemory()

#     def save_long_term(self, user_input: str, ai_output: str):
#         """
#         保存长期记忆（可做策略过滤）
#         """

#         text = f"用户说: {user_input}\nAI回复: {ai_output}"

#         # 👉 可以加策略：只存重要内容
#         if len(user_input) > 20:
#             self.long_term.save_memory(text)

#     def get_long_term_context(self, query: str):
#         docs = self.long_term.search_memory(query)

#         return "\n".join([doc.page_content for doc in docs])