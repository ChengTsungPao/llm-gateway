"""Client protocols besides OpenAI Chat Completions (Adapter pattern).

Internally the gateway speaks one format, OpenAI Chat Completions. Each module here translates
another wire protocol to it and back, so Claude Code and Codex can use local models:

    anthropic.py   Anthropic Messages  (/v1/messages)   used by Claude Code (ANTHROPIC_BASE_URL)
    responses.py   OpenAI Responses    (/v1/responses)  used by Codex (wire_api = "responses")
    stream.py      reads the internal Chat Completions stream as simple events both adapters share
"""
