"""dsh plugin — run DeepSeek Harness agents from Hermes.

Registers one tool, ``dsh_agent``, that spawns
``dsh --profile headless --json "<task>"`` in a working directory, parses the
NDJSON run events, and returns the final answer plus the dsh session id so a
follow-up call can resume the same agent session.

The dsh profile owns the model route; this plugin never passes a model.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_MAX_TEXT = 40_000
_MAX_STDERR = 2_000

DSH_SCHEMA = {
    "name": "dsh_agent",
    "description": (
        "Run a DeepSeek Harness (dsh) agent on a task and return its final answer. "
        "The dsh agent has its own shell and file tools and works inside cwd, on the "
        "model route configured in its dsh profile. Pass session_id from a previous "
        "call to continue that same agent session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the dsh agent should do. It runs its own tools "
                               "(shell, file edits) inside cwd.",
            },
            "cwd": {
                "type": "string",
                "description": "Absolute directory the dsh agent works in. Defaults to "
                               "the current session working directory.",
            },
            "session_id": {
                "type": "string",
                "description": "Resume an existing dsh session (the session_id returned "
                               "by a previous dsh_agent call) instead of starting fresh.",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Hard wall-clock limit for the run. Default 600, max 1800.",
            },
            "profile": {
                "type": "string",
                "description": "dsh profile to boot. Default 'headless'; 'web' is a "
                               "long-running server and is not a valid one-shot target.",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def _find_dsh() -> str | None:
    found = shutil.which("dsh")
    if found:
        return found
    for pattern in ("~/.nvm/versions/node/*/bin/dsh", "/usr/local/bin/dsh", "~/.local/bin/dsh"):
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[-1]
    return None


def _check_available() -> bool:
    return _find_dsh() is not None


def _env_with_keys() -> dict[str, str]:
    """Inherit the environment, then resolve provider keys through Hermes' secret scope.

    The dsh profile names the env var it needs (for example ``OPENCODE_GO_API_KEY``
    or ``DEEPSEEK_API_KEY``); dsh inherits it from the launching environment.
    Hermes credentials live in the active profile's secret scope, so resolve them
    with ``agent.secret_scope.get_secret`` instead of reading ``.env`` files.
    """
    env = dict(os.environ)
    try:
        from agent.secret_scope import get_secret
    except Exception:
        return env
    for name in ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY", "DEEPSEEK_API_KEY"):
        if env.get(name):
            continue
        try:
            value = get_secret(name)
        except Exception:
            continue
        if value:
            env[name] = value
    return env


def _result(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False)


def handle_dsh_agent(args: dict[str, Any], **_kw) -> str:
    task = str(args.get("task") or "").strip()
    if not task:
        return _result(ok=False, error="task is required")

    binary = _find_dsh()
    if not binary:
        return _result(
            ok=False,
            error="dsh is not on PATH. Install it with "
                  "`npm install -g @deepseek-ai/dsh` and retry.",
        )

    cwd = str(args.get("cwd") or os.getcwd())
    if not os.path.isdir(cwd):
        return _result(ok=False, error=f"cwd does not exist: {cwd}")

    profile = str(args.get("profile") or "headless").strip() or "headless"
    if profile == "web":
        return _result(
            ok=False,
            error="profile 'web' serves a browser UI and never exits; use 'headless'.",
        )

    try:
        timeout_s = int(args.get("timeout_s") or 600)
    except (TypeError, ValueError):
        timeout_s = 600
    timeout_s = max(10, min(1800, timeout_s))

    cmd = [binary, "--profile", profile, "--json"]
    session_id = str(args.get("session_id") or "").strip()
    if session_id:
        cmd += ["--session-id", session_id]
    cmd.append(task)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=_env_with_keys(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        code = -1
        timed_out = True
    except OSError as exc:
        return _result(ok=False, error=f"could not start dsh: {exc}")

    duration = round(time.monotonic() - started, 1)

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    session = ""
    final_text = ""
    final_reason: Any = None
    usage: dict[str, Any] = {}
    turns = 0
    parse_errors = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            parse_errors += 1
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "session":
            session = str(event.get("sessionId") or session)
        elif kind == "text":
            texts.append(str(event.get("text") or ""))
        elif kind == "tool_call":
            tool_calls.append({"tool": event.get("tool"), "status": "called"})
        elif kind == "tool_result":
            for call in reversed(tool_calls):
                if call.get("status") == "called":
                    call["status"] = event.get("status") or "completed"
                    break
        elif kind == "status":
            phase = event.get("phase")
            if phase == "turn_start":
                turns += 1
            if phase == "step_end" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
        elif kind == "final":
            session = str(event.get("sessionId") or session)
            final_text = str(event.get("text") or "")
            final_reason = event.get("reason")

    answer = final_text or "".join(texts)
    if len(answer) > _MAX_TEXT:
        answer = answer[:_MAX_TEXT] + f"\n… [truncated {len(answer) - _MAX_TEXT} chars]"

    error = ""
    if timed_out:
        error = f"dsh timed out after {timeout_s}s and was killed."
    elif code != 0 and not final_text:
        error = f"dsh exited {code} without a final message."

    return _result(
        ok=not error,
        error=error or None,
        text=answer,
        session_id=session or None,
        cwd=cwd,
        profile=profile,
        resumed_from=session_id or None,
        exit_code=code,
        timed_out=timed_out,
        duration_s=duration,
        turns=turns,
        tool_calls=tool_calls[:50],
        tool_call_count=len(tool_calls),
        usage=usage or None,
        final_reason=final_reason,
        stderr_tail=(stderr or "")[-_MAX_STDERR:] or None,
        parse_errors=parse_errors or None,
    )


# ─── Persistent SDK sessions ────────────────────────────────────────────────
#
# `dsh --profile sdk` is DeepSeek Harness's automation protocol: newline-
# delimited JSON-RPC over stdio. Unlike the one-shot headless run above, one
# process holds live agent sessions, streams every session event (reasoning,
# tool calls, results), asks the client for tool approvals, and lets the
# working directory change mid-session. Server identity is
# `deepseek-harness-sdk-runtime`; the protocol lives in
# @deepseek-ai/dsh-sdk-protocol.

SDK_SCHEMA = {
    "name": "dsh_session",
    "description": (
        "Control a live DeepSeek Harness (dsh) agent session over dsh's SDK "
        "protocol — the same runtime the dsh UI drives. Actions: start a "
        "session, prompt it (streams its reasoning/tool calls and returns the "
        "answer), resume a persisted session, read or change its working "
        "directory, list live sessions, or shut the runtime down. Use this "
        "instead of dsh_agent when you need multi-turn work, progress detail, "
        "or approvals."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "prompt", "fanout", "resume", "cwd", "list", "shutdown"],
                "description": "start: new session. prompt: send a turn and wait. "
                               "fanout: run several independent dsh agents in parallel. "
                               "resume: reattach a persisted dsh session id. cwd: read "
                               "(no path) or change (path) the session directory. "
                               "list: live sessions. shutdown: stop the runtime.",
            },
            "tasks": {
                "type": "array",
                "description": "For action=fanout: one entry per agent. Each entry is a "
                               "task string, or an object with task and optional cwd.",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "cwd": {"type": "string"},
                    },
                    "required": ["task"],
                },
            },
            "max_parallel": {
                "type": "integer",
                "description": "For action=fanout: how many agents run at once. Default 4, max 8.",
            },
            "session_id": {
                "type": "string",
                "description": "Session for prompt/cwd/resume.",
            },
            "task": {
                "type": "string",
                "description": "Prompt text for action=prompt.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for action=start. Defaults to the "
                               "session working directory.",
            },
            "path": {
                "type": "string",
                "description": "New absolute directory for action=cwd.",
            },
            "provider": {
                "type": "string",
                "description": "Provider route for a new runtime. Default opencode-go.",
            },
            "model": {
                "type": "string",
                "description": "Model for a new runtime. Default deepseek-v4.1-flash.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort for a new runtime. Default max.",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Wall-clock limit for a prompt. Default 600, max 1800.",
            },
            "approval": {
                "type": "string",
                "enum": ["ask", "allow", "reject"],
                "description": "How to answer dsh tool-approval requests during this "
                               "prompt. ask (default): refuse the escalation and return it "
                               "so the user can decide; allow: grant allowed-once; reject: "
                               "refuse. Grants never persist.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

_DEFAULT_PROVIDER = "deepseek-official"
_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_EFFORT = "max"


class _SdkError(RuntimeError):
    pass


class _SdkServer:
    """One `dsh --profile sdk` process and its JSON-RPC plumbing."""

    def __init__(self, provider: str, model: str, reasoning_effort: str, cwd: str):
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.cwd = cwd
        self.approval = "ask"
        self.approvals: list[dict[str, Any]] = []
        self._approval_modes: dict[str, str] = {}
        self._event_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self._pending: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._id = 0
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._status: dict[str, str] = {}
        self._stderr: list[str] = []
        self._dead = False
        self._dead_reason = ""

        binary = _find_dsh()
        if not binary:
            raise _SdkError("dsh is not on PATH")
        self._proc = subprocess.Popen(
            [binary, "--profile", "sdk"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=cwd,
            env=_env_with_keys(),
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._stderr_loop, daemon=True).start()

        info = self.request("initialize", {
            "cwd": cwd,
            "provider": provider,
            "model": model,
            "reasoningEffort": reasoning_effort,
            "approvalRequests": True,
            "capabilities": {"userQuestions": True},
        }, timeout=60)
        name = ((info or {}).get("serverInfo") or {}).get("name")
        if name != "deepseek-harness-sdk-runtime":
            raise _SdkError(f"unexpected SDK server: {name!r}")

    # -- transport ---------------------------------------------------------
    def _read_loop(self) -> None:
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("method") and "id" in msg:
                    self._answer_server_request(msg)
                elif "id" in msg:
                    with self._lock:
                        slot = self._pending.get(msg.get("id"))
                    if slot is not None:
                        slot["result"] = msg
                        slot["event"].set()
                elif msg.get("method"):
                    self._on_notification(msg)
        except Exception as exc:  # pragma: no cover - pipe teardown
            self._mark_dead(f"reader stopped: {exc}")
        finally:
            self._mark_dead("dsh sdk process exited")

    def _stderr_loop(self) -> None:
        try:
            for line in self._proc.stderr:  # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    self._stderr.append(line)
                    del self._stderr[:-40]
        except Exception:
            pass

    def _mark_dead(self, reason: str) -> None:
        if self._dead:
            return
        self._dead = True
        self._dead_reason = reason
        with self._lock:
            slots = list(self._pending.values())
        for slot in slots:
            slot["event"].set()

    def _on_notification(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params") or {}
        sid = str(params.get("sessionId") or "")
        if method == "session.event":
            event = params.get("event") or {}
            if sid:
                with self._event_lock:
                    self._events.setdefault(sid, []).append(event)
        elif method == "session.status":
            if sid:
                with self._event_lock:
                    self._status[sid] = str(params.get("status") or "")

    def _answer_server_request(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if method == "approval/request":
            params = msg.get("params") or {}
            sid = str(params.get("sessionId") or "")
            mode = self._approval_modes.get(sid, "ask")
            if mode == "allow":
                outcome, decision = "allowed-once", "allowed"
            else:
                # "ask" and "reject" both refuse here. "ask" records the request so
                # the model can put it to the user and retry with approval=allow.
                outcome, decision = "rejected", ("needs-user" if mode == "ask" else "rejected")
            with self._event_lock:
                self.approvals.append({
                    "session_id": sid,
                    "tool": params.get("toolName"),
                    "reason": params.get("reason"),
                    "outcome": outcome,
                    "decision": decision,
                })
            result: dict[str, Any] = {"outcome": outcome}
        elif method == "user-question/ask":
            # No interactive user in this path; dismissal is never approval.
            result = {"cancelled": True}
        else:
            result = {}
        try:
            self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except (OSError, ValueError):
            self._mark_dead("stdin closed")

    def request(self, method: str, params: Any, timeout: float = 60) -> Any:
        if self._dead:
            raise _SdkError(f"dsh sdk runtime is not running: {self._dead_reason}")
        slot = {"event": threading.Event(), "result": None}
        with self._lock:
            self._id += 1
            rid = self._id
            self._pending[rid] = slot
        try:
            payload = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                payload["params"] = params
            self._proc.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except (OSError, ValueError) as exc:
            with self._lock:
                self._pending.pop(rid, None)
            self._mark_dead(f"stdin closed: {exc}")
            raise _SdkError(f"could not send {method}: {exc}") from exc

        if not slot["event"].wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise _SdkError(f"{method} timed out after {timeout}s")
        with self._lock:
            self._pending.pop(rid, None)
        msg = slot["result"] or {}
        if "error" in msg:
            err = msg["error"] or {}
            raise _SdkError(str(err.get("message") or err))
        return msg.get("result")

    # -- sessions ----------------------------------------------------------
    def open_session(self, session_id: str, cwd: str) -> None:
        self._events.setdefault(session_id, [])
        self._status[session_id] = "idle"
        if cwd and os.path.abspath(cwd) != os.path.abspath(self.cwd):
            self.request("session/working-directory/set",
                         {"sessionId": session_id, "path": os.path.abspath(cwd)}, timeout=30)

    def resume_session(self, session_id: str) -> None:
        self.request("session/resume", {"sessionId": session_id}, timeout=60)
        self._events.setdefault(session_id, [])
        self._status.setdefault(session_id, "idle")

    def set_cwd(self, session_id: str, path: str) -> str:
        res = self.request("session/working-directory/set",
                           {"sessionId": session_id, "path": os.path.abspath(path)}, timeout=30)
        return str((res or {}).get("cwd") or "")

    def get_cwd(self, session_id: str) -> str:
        res = self.request("session/working-directory/get",
                           {"sessionId": session_id}, timeout=30)
        return str((res or {}).get("cwd") or "")

    def prompt(self, session_id: str, text: str, timeout: float, approval: str = "ask") -> dict[str, Any]:
        with self._session_locks_guard:
            session_lock = self._session_locks.setdefault(session_id, threading.Lock())
        with session_lock:
            self._approval_modes[session_id] = approval
            with self._event_lock:
                self.approvals = [a for a in self.approvals if a.get("session_id") != session_id]
                before = len(self._events.setdefault(session_id, []))
            receipt = self.request("session/prompt", {
                "sessionId": session_id,
                "contentBlocks": [{"type": "text", "text": text}],
            }, timeout=60)
            message_id = str((receipt or {}).get("messageId") or "")

            deadline = time.monotonic() + timeout
            user_seq = None
            timed_out = False
            while True:
                with self._event_lock:
                    events = list(self._events.get(session_id, [])[before:])
                if user_seq is None:
                    for ev in events:
                        data = ev.get("data") or {}
                        if ev.get("type") == "user/message" and (
                                not message_id or data.get("id") == message_id):
                            user_seq = int(ev.get("seq") or 0)
                            break
                finished = False
                if user_seq is not None:
                    for ev in events:
                        if ev.get("type") == "turn/end" and int(ev.get("seq") or 0) > user_seq:
                            finished = True
                            break
                if finished:
                    break
                if self._dead:
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                time.sleep(0.25)

            return self._summarize(session_id, before, timed_out)

    def _summarize(self, session_id: str, before: int, timed_out: bool) -> dict[str, Any]:
        with self._event_lock:
            events = list(self._events.get(session_id, [])[before:])
            approvals = [a for a in self.approvals if a.get("session_id") == session_id]
        answer = ""
        reasoning_chars = 0
        tool_calls: list[dict[str, Any]] = []
        turns = 0
        for ev in events:
            kind = ev.get("type")
            data = ev.get("data") or {}
            if kind == "assistant/message":
                message = data.get("message") or {}
                parts = []
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif block.get("type") == "reasoning":
                        reasoning_chars += len(str(block.get("text") or ""))
                if parts:
                    answer = "".join(parts)
            elif kind == "tool/call":
                tool_calls.append({
                    "tool": data.get("name"),
                    "arguments": str(data.get("arguments") or "")[:300],
                })
            elif kind == "turn/end":
                turns += 1
        if len(answer) > _MAX_TEXT:
            answer = answer[:_MAX_TEXT] + f"\n… [truncated {len(answer) - _MAX_TEXT} chars]"
        pending = [a for a in approvals if a.get("decision") == "needs-user"]
        guidance = None
        if pending:
            tools = ", ".join(sorted({str(a.get("tool")) for a in pending}))
            guidance = (
                f"dsh asked to escalate for {tools}. This call answered rejected because "
                "approval=ask, so the operation did not run. Put the request to the user; "
                "if they agree, call dsh_session again with approval='allow'."
            )
        return {
            "text": answer,
            "session_id": session_id,
            "status": self._status.get(session_id),
            "timed_out": timed_out,
            "turns": turns,
            "tool_calls": tool_calls[:50],
            "tool_call_count": len(tool_calls),
            "reasoning_chars": reasoning_chars,
            "approvals": approvals,
            "pending_approvals": pending,
            "guidance": guidance,
            "events": len(events),
            "cwd": self.cwd,
            "provider": self.provider,
            "model": self.model,
            "stderr_tail": "\n".join(self._stderr[-10:]) or None,
        }

    def shutdown(self) -> None:
        try:
            self.request("shutdown", None, timeout=15)
        except _SdkError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._mark_dead("shut down")
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass


# provider/model/effort -> server; session id -> session record
_SERVERS: dict[tuple[str, str, str], _SdkServer] = {}
_SESSIONS: dict[str, dict[str, Any]] = {}
_SDK_LOCK = threading.Lock()


def _server_for(provider: str, model: str, effort: str, cwd: str) -> _SdkServer:
    key = (provider, model, effort)
    server = _SERVERS.get(key)
    if server is None or server._dead:
        server = _SdkServer(provider, model, effort, cwd)
        _SERVERS[key] = server
    return server


def _prompt_result(session_id: str, task: str, timeout_s: int, approval: str) -> dict[str, Any]:
    with _SDK_LOCK:
        record = _SESSIONS.get(session_id)
    if record is None:
        return {"ok": False, "session_id": session_id,
                "error": f"unknown session {session_id!r}; start or resume it first"}
    server = _SERVERS.get(tuple(record["server"]))
    if server is None or server._dead:
        return {"ok": False, "session_id": session_id,
                "error": "dsh runtime for this session is gone; start a new session"}
    summary = server.prompt(session_id, task, timeout_s, approval)
    record["cwd"] = summary.get("cwd") or record["cwd"]
    return {"ok": not summary.get("timed_out"),
            "error": "timed out; partial result" if summary.get("timed_out") else None,
            "action": "prompt", **summary}


def _new_session(provider: str, model: str, effort: str, cwd: str) -> dict[str, Any]:
    server = _server_for(provider, model, effort, cwd)
    session_id = f"hermes-{uuid.uuid4()}"
    server.open_session(session_id, cwd)
    with _SDK_LOCK:
        _SESSIONS[session_id] = {"server": (provider, model, effort), "cwd": cwd,
                                 "created": time.time()}
    return {"session_id": session_id, "cwd": server.get_cwd(session_id) or cwd}


def _timeout_arg(args: dict[str, Any]) -> int:
    try:
        value = int(args.get("timeout_s") or 600)
    except (TypeError, ValueError):
        value = 600
    return max(10, min(1800, value))


def _approval_arg(args: dict[str, Any]) -> str:
    value = str(args.get("approval") or "ask").strip().lower()
    return value if value in {"ask", "allow", "reject"} else "ask"


def handle_dsh_session(args: dict[str, Any], **_kw) -> str:
    action = str(args.get("action") or "").strip().lower()
    provider = str(args.get("provider") or _DEFAULT_PROVIDER)
    model = str(args.get("model") or _DEFAULT_MODEL)
    effort = str(args.get("reasoning_effort") or _DEFAULT_EFFORT)
    cwd = str(args.get("cwd") or os.getcwd())
    session_id = str(args.get("session_id") or "").strip()

    try:
        if action == "start":
            if not os.path.isdir(cwd):
                return _result(ok=False, error=f"cwd does not exist: {cwd}")
            info = _new_session(provider, model, effort, cwd)
            return _result(ok=True, action="start", provider=provider, model=model,
                           reasoning_effort=effort, **info)

        if action == "prompt":
            if not session_id:
                return _result(ok=False, error="session_id is required for prompt")
            task = str(args.get("task") or "").strip()
            if not task:
                return _result(ok=False, error="task is required for prompt")
            return _result(**_prompt_result(session_id, task, _timeout_arg(args), _approval_arg(args)))

        if action == "fanout":
            tasks = args.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                return _result(ok=False, error="tasks is required for fanout (a non-empty list)")
            prepared: list[tuple[str, str]] = []
            for index, item in enumerate(tasks):
                if isinstance(item, str):
                    task_text, item_cwd = item, cwd
                elif isinstance(item, dict):
                    task_text = str(item.get("task") or "")
                    item_cwd = str(item.get("cwd") or cwd)
                else:
                    return _result(ok=False, error=f"tasks[{index}] must be a string or object")
                if not task_text.strip():
                    return _result(ok=False, error=f"tasks[{index}] has an empty task")
                if not os.path.isdir(item_cwd):
                    return _result(ok=False, error=f"tasks[{index}] cwd does not exist: {item_cwd}")
                info = _new_session(provider, model, effort, item_cwd)
                prepared.append((info["session_id"], task_text.strip()))
            try:
                max_parallel = int(args.get("max_parallel") or 4)
            except (TypeError, ValueError):
                max_parallel = 4
            max_parallel = max(1, min(8, max_parallel))
            timeout_s, approval = _timeout_arg(args), _approval_arg(args)
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(max_parallel, len(prepared))) as pool:
                futures = [pool.submit(_prompt_result, sid, task, timeout_s, approval)
                           for sid, task in prepared]
                for (sid, _task), future in zip(prepared, futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive
                        results.append({"ok": False, "session_id": sid,
                                        "error": f"{type(exc).__name__}: {exc}"})
            return _result(ok=all(r.get("ok") for r in results), action="fanout",
                           results=results, agents=len(results),
                           failed=sum(1 for r in results if not r.get("ok")),
                           max_parallel=max_parallel, provider=provider, model=model)

        if action == "resume":
            if not session_id:
                return _result(ok=False, error="session_id is required for resume")
            server = _server_for(provider, model, effort, cwd)
            server.resume_session(session_id)
            with _SDK_LOCK:
                _SESSIONS[session_id] = {"server": (provider, model, effort), "cwd": cwd,
                                         "created": time.time()}
            return _result(ok=True, action="resume", session_id=session_id,
                           cwd=server.get_cwd(session_id) or cwd)

        if action == "cwd":
            if not session_id:
                return _result(ok=False, error="session_id is required for cwd")
            with _SDK_LOCK:
                record = _SESSIONS.get(session_id)
            if record is None:
                return _result(ok=False, error=f"unknown session {session_id!r}")
            server = _SERVERS.get(tuple(record["server"]))
            if server is None or server._dead:
                return _result(ok=False, error="dsh runtime for this session is gone")
            path = str(args.get("path") or "").strip()
            if path:
                new_cwd = server.set_cwd(session_id, path)
                record["cwd"] = new_cwd or record["cwd"]
                return _result(ok=True, action="cwd", session_id=session_id, cwd=new_cwd)
            return _result(ok=True, action="cwd", session_id=session_id,
                           cwd=server.get_cwd(session_id))

        if action == "list":
            live = []
            with _SDK_LOCK:
                sessions = list(_SESSIONS.items())
                runtimes = len(_SERVERS)
            for sid, record in sessions:
                server = _SERVERS.get(tuple(record["server"]))
                live.append({
                    "session_id": sid,
                    "cwd": record["cwd"],
                    "provider": record["server"][0],
                    "model": record["server"][1],
                    "status": (server._status.get(sid) if server else "gone"),
                    "age_s": round(time.time() - record["created"], 1),
                })
            return _result(ok=True, action="list", sessions=live, runtimes=runtimes)

        if action == "shutdown":
            with _SDK_LOCK:
                servers = list(_SERVERS.values())
            closed = 0
            for server in servers:
                server.shutdown()
                closed += 1
            with _SDK_LOCK:
                _SERVERS.clear()
                _SESSIONS.clear()
            return _result(ok=True, action="shutdown", runtimes_closed=closed)

        return _result(ok=False, error=f"unknown action {action!r}")
    except _SdkError as exc:
        return _result(ok=False, error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return _result(ok=False, error=f"{type(exc).__name__}: {exc}")


def register(ctx) -> None:
    global _DEFAULT_PROVIDER, _DEFAULT_MODEL, _DEFAULT_EFFORT
    _DEFAULT_PROVIDER = str(ctx.get_config("route.provider", _DEFAULT_PROVIDER) or _DEFAULT_PROVIDER)
    _DEFAULT_MODEL = str(ctx.get_config("route.model", _DEFAULT_MODEL) or _DEFAULT_MODEL)
    _DEFAULT_EFFORT = str(ctx.get_config("route.reasoning_effort", _DEFAULT_EFFORT) or _DEFAULT_EFFORT)
    ctx.register_tool(
        name="dsh_agent",
        toolset="dsh",
        schema=DSH_SCHEMA,
        handler=handle_dsh_agent,
        check_fn=_check_available,
        emoji="🐋",
        description="Run a one-shot DeepSeek Harness (dsh) agent task and return its answer.",
    )
    ctx.register_tool(
        name="dsh_session",
        toolset="dsh",
        schema=SDK_SCHEMA,
        handler=handle_dsh_session,
        check_fn=_check_available,
        emoji="🌊",
        description="Drive a live DeepSeek Harness agent session (start/prompt/resume/cwd/list/shutdown).",
    )
