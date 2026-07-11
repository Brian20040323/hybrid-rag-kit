# ReAct 智能体

ReAct 通过 Thought、Action、Observation 循环让模型决定何时调用工具。
它比单次 function calling 更适合需要多步推理与可观测轨迹的任务。
常见护栏包括最大步数、重复动作熔断、工具错误预算。
