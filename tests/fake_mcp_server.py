"""A tiny MCP server over stdio for the tests: one read-only tool `add` and one failing tool."""

import json
import sys

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue  # notification
    method, rid = msg["method"], msg["id"]
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "add", "description": "Add two numbers", "annotations": {"readOnlyHint": True},
             "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                             "required": ["a", "b"]}},
            {"name": "explode", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
        ]}
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"]["arguments"]
        if name == "add":
            result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}
        else:
            result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method"}}),
              flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)
