#!/usr/bin/env python3
"""Fake dsh CLI for offline tests.

Speaks the same shapes the plugin parses:

  dsh --profile headless --json <task>   -> NDJSON run events
  dsh --profile sdk                      -> newline-delimited JSON-RPC

Set the task to FAIL for a non-zero headless exit, or ESCALATE to make the SDK
runtime send an approval/request before finishing the turn.
"""
import json
import sys


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def headless(task):
    out({"type": "session", "sessionId": "fake-session-1", "cwd": "."})
    out({"type": "status", "phase": "turn_start", "turn": 1})
    if "FAIL" in task:
        sys.stderr.write("fake dsh: deliberate failure\n")
        return 3
    out({"type": "text", "text": "FAKE_OK"})
    out({"type": "status", "phase": "step_end", "turn": 1, "step": 1,
         "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}})
    out({"type": "status", "phase": "turn_end", "turn": 1,
         "reason": {"kind": "completed"}})
    out({"type": "final", "sessionId": "fake-session-1",
         "reason": {"kind": "completed"}, "text": "FAKE_OK"})
    return 0


def sdk():
    cwd = "/tmp"
    pending_approval = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "method" not in msg:  # the client answering our approval request
            if msg.get("id") == 900:
                pending_approval = (msg.get("result") or {}).get("outcome")
            continue
        mid, method = msg.get("id"), msg.get("method")
        params = msg.get("params") or {}

        if method == "initialize":
            cwd = params.get("cwd") or cwd
            out({"jsonrpc": "2.0", "id": mid, "result": {
                "serverInfo": {"name": "deepseek-harness-sdk-runtime", "version": "0.0.1"}}})
        elif method == "session/prompt":
            sid = params.get("sessionId")
            text = "".join(b.get("text", "") for b in params.get("contentBlocks") or [])
            message_id = "m-1"
            out({"jsonrpc": "2.0", "id": mid, "result": {"messageId": message_id}})
            out({"jsonrpc": "2.0", "method": "session.event", "params": {
                "sessionId": sid,
                "event": {"type": "user/message", "seq": 1,
                          "data": {"id": message_id,
                                   "content": [{"type": "text", "text": text}]}}}})
            if "ESCALATE" in text:
                out({"jsonrpc": "2.0", "id": 900, "method": "approval/request",
                     "params": {"sessionId": sid, "toolName": "write",
                                "reason": "outside workspace"}})
                answer = sys.stdin.readline()
                if answer.strip():
                    pending_approval = (json.loads(answer).get("result") or {}).get("outcome")
            question_cancelled = None
            if "QUESTION" in text:
                out({"jsonrpc": "2.0", "id": 901, "method": "user-question/ask",
                     "params": {"requestId": "q-1", "sessionId": sid,
                                "questions": [{"id": "q-1", "question": "Which branch?"}]}})
                answer = sys.stdin.readline()
                if answer.strip():
                    question_cancelled = (json.loads(answer).get("result") or {}).get("cancelled")
            answer_text = "FAKE_OK"
            if pending_approval:
                answer_text += f" approval={pending_approval}"
            if question_cancelled:
                answer_text += " question=cancelled"
            out({"jsonrpc": "2.0", "method": "session.event", "params": {
                "sessionId": sid,
                "event": {"type": "assistant/message", "seq": 2,
                          "data": {"message": {"role": "assistant", "content": [
                              {"type": "text", "text": answer_text}]}}}}})
            out({"jsonrpc": "2.0", "method": "session.event", "params": {
                "sessionId": sid,
                "event": {"type": "turn/end", "seq": 3,
                          "data": {"turn": 1, "reason": {"kind": "completed"}}}}})
            out({"jsonrpc": "2.0", "method": "session.status",
                 "params": {"sessionId": sid, "status": "idle"}})
        elif method == "session/resume":
            out({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "session/working-directory/get":
            out({"jsonrpc": "2.0", "id": mid, "result": {"cwd": cwd}})
        elif method == "session/working-directory/set":
            cwd = params.get("path") or cwd
            out({"jsonrpc": "2.0", "id": mid, "result": {"cwd": cwd}})
        elif method == "shutdown":
            out({"jsonrpc": "2.0", "id": mid, "result": {}})
            return 0
        else:
            out({"jsonrpc": "2.0", "id": mid,
                 "error": {"code": -32601, "message": f"unknown method {method}"}})
    return 0


def main():
    args = sys.argv[1:]
    if "--profile" not in args:
        sys.stderr.write("fake dsh: --profile is required\n")
        return 2
    profile = args[args.index("--profile") + 1]
    if profile == "headless":
        task = args[-1] if args else ""
        return headless(task)
    if profile == "sdk":
        return sdk()
    sys.stderr.write(f"fake dsh: unsupported profile {profile}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
