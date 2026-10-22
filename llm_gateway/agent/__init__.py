"""A small coding-agent harness (in the spirit of Claude Code) on top of the gateway.

Read docs/agent.md first; it maps every module to the agent-loop diagram. In short:

    loop.py         the agent loop: two states, Call LLM <-> Run tools (ReAct)
    llm.py          talks to any OpenAI-compatible endpoint (here: the gateway), streaming
    tools/          the tools; each one is a small class (Command pattern) in a registry
    errors.py       unified tool-error format + failure classification, retry, circuit breaker
    permissions.py  permission modes and the approval strategy (human in the loop)
    hooks.py        user-configured shell hooks at fixed lifecycle events
    context.py      system prompt (memory files, skills index) and context-window management
    mcp.py          a minimal MCP client; remote tools are adapted into local Tool objects
    session.py      conversation state, checkpointed to disk after every step
    events.py       what the loop reports to a UI (Observer): CLI and web render the same events
    service.py      runs agents for the web UI (one background task per turn + event queue)
    hosting.py      where web agents run (Strategy): in this process, or ...
    k8s.py          ... each in its own Kubernetes pod with its own folder
    podserver.py    the program inside an agent pod
    cli.py          the `llm-agent` terminal UI
"""
