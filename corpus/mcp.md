# MCP 协议

Model Context Protocol 用 JSON-RPC 暴露 tools/list、tools/call，以及 resources。
IDE 与 Agent 可以标准方式挂载外部知识库与工具，而无需为每个客户端写适配器。
知识桥可以把本地 KB 检索包装成 MCP tool，让 Cursor / Agent 统一调用。
