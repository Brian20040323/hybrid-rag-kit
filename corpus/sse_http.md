# SSE 流式接口工程要点

SSE（Server-Sent Events）适合把 Agent 中间步骤推送给前端。
与 WebSocket 不同，SSE 偏单向推送，实现更轻。
本文聚焦 HTTP 流式传输与浏览器 EventSource，不讨论 Trace spans 或 Tool Call 耗时观测模型。
若查询包含 SSE 但目标是可观测性，需要区分工程传输层与 Agent Trace。
