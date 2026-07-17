# 纯 Prompt Agent 的局限

仅靠 Prompt 让大模型“假装调用工具”，无法真正访问私有知识库或实时网页。
幻觉回复常见：编造制度条款、伪造链接、忽略权限边界。
真正的 Agent 需要 Function Calling / ReAct 工具闭环，而不是更长的系统提示词。
本文也出现 Agent、工具、知识库等词，但并不讨论 BM25、RRF 或混合检索实现细节。
