<div align="center">

<h1>hermes-dsh</h1>

<p><strong>Drive DeepSeek Harness agents from Hermes.</strong></p>

<p>One-shot runs, live sessions, and parallel agent fanout, with dsh's own tools,<br>
sandbox, approvals, and session state.</p>

[![tests](https://github.com/haroon0x/hermes-dsh/actions/workflows/test.yml/badge.svg)](https://github.com/haroon0x/hermes-dsh/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![hermes](https://img.shields.io/badge/hermes-%3E%3D0.22-6f42c1.svg)](https://github.com/NousResearch/hermes-agent)
[![dsh](https://img.shields.io/badge/dsh-0.2.x-black.svg)](https://github.com/deepseek-ai/deepseek-harness)

</div>

Hermes already talks to DeepSeek as a model provider. That is not the same thing
as running dsh's agent loop. This plugin drives [DeepSeek
Harness](https://github.com/deepseek-ai/deepseek-harness) itself, so the agent
that runs your task has dsh's tools, dsh's sandbox, dsh's approval prompts, and
dsh's session state.

Two surfaces:

| | transport | use it for |
| --- | --- | --- |
| `dsh_agent` | `dsh --profile headless --json` | fire and forget, no live control |
| `dsh_session` | `dsh --profile sdk` (JSON-RPC over stdio) | multi-turn work, approvals, parallel agents |

## Install

```bash
hermes plugins install haroon0x/hermes-dsh
hermes plugins enable dsh
```

From a clone instead:

```bash
git clone https://github.com/haroon0x/hermes-dsh ~/.hermes/plugins/dsh
hermes plugins enable dsh
```

The catalog entry is in review ([NousResearch/hermes-agent#157323](https://github.com/NousResearch/hermes-agent/pull/157323)).

## 30 seconds

```
> dsh_session(action="start", cwd="/repo")
  session_id: hermes-1c75d9c6-bc95-46b6-8102-6c9d0092c6b2
  cwd:        /repo

> dsh_session(action="prompt", session_id="hermes-1c75...", task="find the flaky test")
  text:        tests/test_billing.py::test_retry. It sleeps 50ms then asserts on
               wall-clock timing; it fails under load. Use a fake clock.
  tool_calls:  grep, read, bash
  turns:       3
  approvals:   []
```

Or run three agents at once:

```
> dsh_session(action="fanout", cwd="/repo", max_parallel=3,
              tasks=["audit auth", "audit payments", "audit uploads"])
  agents: 3  failed: 0
  - hermes-2844...  auth: session tokens are not rotated on privilege change
  - hermes-cfae...  payments: refund path trusts a client-supplied amount
  - hermes-1f02...  uploads: extension check runs before the file is decoded
```

Every result keeps its `session_id`. You can prompt that agent again, ask it
follow-up questions, or feed its answer into another agent.

## What you get

- **Two modes.** A one-shot run when you just want the answer, a live session
  when you need turns, approvals, or control.
- **Parallel fanout.** Up to 8 agents per call, each with its own working
  directory, and optionally its own provider/model.
- **Approvals that stay honest.** `ask` is the default: the plugin refuses the
  escalation, returns `pending_approvals`, and lets the model put it to you.
  Nothing is ever auto-approved.
- **User questions surfaced, not swallowed.** dsh's `ask_user_question` is
  answered `cancelled` and returned in `pending_questions`, so the model asks
  you and sends your answer back.
- **Progress you can watch.** `events` with a `since_seq` cursor reads dsh's
  stream while a turn is still running.
- **No leaked processes.** Idle runtimes stop after 15 minutes, and `atexit`
  cleans up when Hermes exits.
- **Resumable sessions.** Keep the dsh session id and `resume` it after a
  restart.
- **Offline tests.** 21 tests against a fake `dsh` binary, no key needed.

## dsh_agent

```python
dsh_agent(
    task="run the tests and summarise failures",
    cwd="/abs/project",
    timeout_s=900,
)
```

Returns `text`, `session_id`, `tool_calls`, `tool_call_count`, `usage`,
`duration_s`, `exit_code`, and `stderr_tail`. Pass `session_id` back to continue
the same persisted dsh session. On timeout the child is killed and you get the
partial answer with `timed_out: true`.

The `headless` profile owns the model route. This tool does not pick a model.

## dsh_session

| action | what it does |
| --- | --- |
| `start` | new session in `cwd`, returns `session_id` |
| `prompt` | send a turn, wait for `turn/end`, return answer and tool calls |
| `fanout` | run several independent agents concurrently |
| `events` | read the session's event stream since `since_seq` |
| `resume` | reattach a persisted dsh session id |
| `cwd` | read the session directory, or change it with `path` |
| `list` | live sessions, their route, cwd, status, and age |
| `shutdown` | stop every runtime this plugin started |

A prompt returns the last assistant text, every tool call (name plus a truncated
argument string), reasoning character count, event count, turn count, and the
approval decisions for that session.

### Approvals

dsh asks before it steps outside its sandbox. `approval` decides the answer:

| value | behaviour |
| --- | --- |
| `ask` (default) | refuse, record in `pending_approvals`, return guidance |
| `allow` | grant `allowed-once` for that one operation |
| `reject` | refuse |

Grants never persist past the operation. The plugin never approves on its own,
and an approval in one session never covers another session.

### User questions

`ask_user_question` goes to whoever drives the runtime, which here is Hermes,
not a person. The plugin answers `{cancelled: true}` (a dismissal is not
permission), returns the question in `pending_questions`, and the model asks
you. Your answer goes back as the next prompt on that session.

## Fanout

```python
dsh_session(
    action="fanout",
    cwd="/repo",
    max_parallel=4,
    tasks=[
        "audit auth middleware",
        "audit payment flow",
        {"task": "audit uploads", "cwd": "/repo/uploads"},
        {"task": "review the auth diff", "model": "deepseek-reasoner"},
    ],
)
```

`max_parallel` defaults to 4 and caps at 8. Entries are strings or objects with
`task`, `cwd`, `provider`, `model`, and `reasoning_effort`. The last two let one
fanout mix a cheap model for bulk work with a stronger one for review.

Results come back in task order, each with `ok`, `session_id`, `text`,
`tool_calls`, `turns`, and `approvals`. Failures do not abort the batch: you get
`failed: n` and the per-agent error.

## Watching progress

`prompt` blocks until `turn/end`, which is fine for short tasks and useless for
long ones. Poll instead:

```
dsh_session(action="events", session_id=..., since_seq=0)
  events:   [turn/start, step/start, tool/call, ...]
  next_seq: 42
  status:   working

dsh_session(action="events", session_id=..., since_seq=42)
  events:   []
  next_seq: 42
```

Event types are dsh's own: `turn/start`, `step/start`, `tool/call`,
`tool/result`, `assistant/message`, `turn/end`. The cursor is monotonic per
session, so passing `next_seq` never replays.

## Configuration

```yaml
plugins:
  entries:
    dsh:
      settings:
        route:
          provider: deepseek-official   # default
          model: deepseek-chat          # default
          reasoning_effort: max         # default
        idle_timeout_s: 900             # 0 disables the idle reaper
```

`route` is what `dsh_session` asks the SDK runtime to use. The `headless`
profile owns its own route; this setting does not change it.

Credentials are never stored or read from files by the plugin. It passes the
environment through to dsh and resolves provider keys with
`agent.secret_scope.get_secret` (`DEEPSEEK_API_KEY`, `OPENCODE_GO_API_KEY`,
`OPENCODE_API_KEY`), so a multi-profile gateway does not leak another profile's
key.

## Requirements

- `dsh` on `PATH`: `npm install -g @deepseek-ai/dsh`
- Hermes `>= 0.22`, Python `3.10+`, Linux or macOS
- a dsh profile that resolves to a provider with a key

The `headless` profile and the `sdk` profile both need the provider adapter
mounted in `~/.dsh/profiles/<profile>/cordis.patch.yml`. If the SDK runtime
fails with `no adapter registered for provider`, that file is missing the
`llm-pi-ai` provider block. This plugin does not write dsh profiles.

## Design

```
Hermes tool call
   |
   +-- dsh_agent  ---> dsh --profile headless --json "task"
   |                    NDJSON events on stdout, one process per run
   |
   +-- dsh_session --> dsh --profile sdk
                        newline-delimited JSON-RPC on stdio
                        server: deepseek-harness-sdk-runtime
                        sessions: N per runtime, concurrent
                        routes:   one runtime per provider/model/effort
```

Why the SDK instead of `dsh web`? The web profile is a long-running browser UI;
its `/api` is that UI's transport, behind a browser-trust fence, and it never
exits, so it cannot be a one-shot target. The SDK is dsh's documented automation
surface and the same runtime the UI drives.

## Limitations

- The SDK has no per-turn cancel. A prompt that times out keeps running inside
  dsh; only `shutdown` stops it, and that stops every session on the runtime.
- Sessions live in the runtime process. Across a Hermes restart, only `resume`
  with a persisted dsh session id brings one back.
- One runtime per route. Sessions on that route share it and run concurrently,
  up to `max_parallel` per fanout call.
- `dsh --profile web` is not a valid target, for the reason above.

## What it runs

The plugin starts `dsh` subprocesses and speaks to them over stdio. It makes no
network calls of its own, writes no files outside dsh's own state, and stores no
credentials. A run can execute shell commands and edit files in the session's
working directory, subject to dsh's permission preset and the approval rules
above.

## Development

```bash
python3 -m unittest discover -s tests -v
hermes plugins validate .
```

The tests use a fake `dsh` executable (`tests/fake_dsh.py`) that emits the same
NDJSON and JSON-RPC shapes as the real CLI, so they run offline without dsh or a
DeepSeek key. CI runs them on Ubuntu and macOS against Python 3.10 and 3.13.

## License

MIT
