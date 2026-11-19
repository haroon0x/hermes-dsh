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
| `dsh_session` | `dsh --profile acp` by default | multi-turn work, cancel, approvals, model config, parallel agents |

`dsh_session` speaks ACP, the same protocol the dsh UI uses. That buys real
per-turn cancel, permission requests, and the session's own model and reasoning
catalog. Pass `transport="sdk"` to use the older JSON-RPC automation runtime
instead; it has no cancel.

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
The `owner/repo` form above works today because it installs straight from GitHub;
the bare name `hermes plugins install dsh` only resolves once that entry merges.

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

- **Two modes.** A one-shot run when you just want the answer, a live ACP
  session when you need turns, cancel, approvals, or control.
- **Parallel fanout.** Up to 8 agents per call, each with its own working
  directory, and optionally its own provider/model.
- **Cancel that works.** `cancel` sends ACP `session/cancel`; the running turn
  stops and returns `stop_reason: cancelled`. Timeouts cancel too.
- **Approvals that stay honest.** `ask` is the default: the plugin refuses the
  escalation and returns `pending_approvals` for the model to put to you.
  `allow` routes through Hermes' approval prompt first, so a person confirms it
  and cron/unattended runs fail closed.
- **Model config from the session.** `config` reads the provider/model/effort
  catalog dsh advertises instead of hardcoding a list.
- **User questions surfaced, not swallowed.** On the SDK transport,
  `ask_user_question` is answered `cancelled` and returned in
  `pending_questions`, so the model asks you and sends your answer back.
- **Progress you can watch.** `events` with a `since_seq` cursor reads dsh's
  stream while a turn is still running.
- **No leaked processes.** Idle runtimes stop after 15 minutes, and `atexit`
  cleans up when Hermes exits.
- **Resumable sessions.** Keep the dsh session id and `resume` it after a
  restart.
- **Offline tests.** 42 tests against a fake `dsh` binary, no key needed.

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
| `start` | new session in `cwd`, applies the route, returns `session_id` and config |
| `prompt` | send a turn, wait for the stop reason, return text, tools and usage |
| `fanout` | run several independent agents concurrently |
| `events` | read the session's update stream since `since_seq` |
| `cancel` | stop the running turn (ACP only) |
| `resume` | reattach a persisted dsh session id |
| `cwd` | read the session directory (ACP binds it at creation) |
| `config` | read the model and reasoning catalog, or set one option |
| `list` | live sessions, their route, cwd, status, and update count |
| `shutdown` | stop every runtime this plugin started |

A prompt returns the last assistant text, every tool call (title, kind, status,
arguments), reasoning character count, update count, usage, and the approval
decisions for that session. On ACP a timed-out prompt sends `session/cancel`
before returning, so the turn actually stops.

### Approvals

dsh asks before it steps outside its sandbox. `approval` decides the answer:

| value | behaviour |
| --- | --- |
| `ask` (default) | refuse, record in `pending_approvals`, return guidance |
| `allow` | grant once, after Hermes asks you to approve the call (denied in cron/unattended runs) |
| `reject` | refuse |

Grants never persist past the operation. The model can set `approval`, but it
cannot hand itself a dsh escalation: a `dsh_session` call with `approval="allow"`
goes through the plugin's `pre_tool_call` hook, which escalates it to Hermes'
human-approval gate (once/session/always/deny, timeout fails closed). Cron
honors `approvals.cron_mode`, and any other unattended context without an
approval bridge blocks the call. An approval in one session never covers
another session.

### User questions

On the SDK runtime, `ask_user_question` goes to whoever drives it, which here is
Hermes, not a person. The plugin answers `{cancelled: true}` (a dismissal is not
permission), returns the question in `pending_questions`, and the model asks
you. Your answer goes back as the next prompt on that session. The ACP runtime
has no equivalent request in this dsh build, so `pending_questions` is empty
there.

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
`tool_calls`, `stop_reason`, `usage`, and `approvals`. Failures do not abort the
batch: you get `failed: n` and the per-agent error.

## Watching progress

`prompt` blocks until the turn stops, which is fine for short tasks and useless
for long ones. Poll instead:

```
dsh_session(action="events", session_id=..., since_seq=0)
  events:   [agent_thought_chunk, tool_call, tool_call_update, ...]
  next_seq: 42
  status:   working

dsh_session(action="events", session_id=..., since_seq=42)
  events:   []
  next_seq: 42
```

ACP updates are dsh's own: `agent_thought_chunk`, `agent_message_chunk`,
`tool_call`, `tool_call_update`, `usage_update`. The SDK transport streams its
own event names (`turn/start`, `step/start`, `tool/call`, `turn/end`). The
cursor is monotonic per session, so passing `next_seq` never replays.

## Cancel

```
dsh_session(action="cancel", session_id=...)
```

Sends ACP `session/cancel`; the in-flight `prompt` returns
`stop_reason: "cancelled"` instead of running to completion. A timeout in
`prompt` does the same thing automatically before returning partial results.

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

`route` is what `dsh_session` asks the ACP session to use, by setting the
session's model and reasoning options at start. The `headless` profile owns its
own route; this setting does not change it.

Credentials are never stored or read from files by the plugin. It builds the
dsh child environment from an explicit allowlist instead of inheriting the
launcher's: PATH, HOME, shell/terminal/locale settings, temp dir, `NODE_*`,
`XDG_*`, `DSH_*` and proxy variables. Provider keys are resolved with
`agent.secret_scope.get_secret` (`DEEPSEEK_API_KEY`, `OPENCODE_GO_API_KEY`,
`OPENCODE_API_KEY`) and only those configured key variables cross into the
child, so a multi-profile gateway does not leak another profile's key or any
other secret Hermes holds.

## Requirements

- `dsh` on `PATH`: `npm install -g @deepseek-ai/dsh`
- Hermes `>= 0.22`, Python `3.10+`, Linux or macOS
- a dsh profile that resolves to a provider with a key

The `headless`, `acp`, and `sdk` profiles each need the provider adapter mounted
in `~/.dsh/profiles/<profile>/cordis.patch.yml`. If a runtime fails with
`no adapter registered for provider` or `no API key for provider route`, that
file is missing the `llm-pi-ai` provider block. This plugin does not write dsh
profiles.

## Design

```
Hermes tool call
   |
   +-- dsh_agent  ---> dsh --profile headless --json "task"
   |                    NDJSON events on stdout, one process per run
   |
   +-- dsh_session --> dsh --profile acp          (default)
   |                    ACP over newline-delimited JSON-RPC
   |                    session/new, prompt, cancel, request_permission
   |                    one runtime, many sessions, model per session
   |
   +-- dsh_session --> dsh --profile sdk          (transport="sdk")
                        JSON-RPC automation runtime
                        no cancel, one runtime per route
```

Why ACP? It is the surface the dsh UI itself drives: per-turn cancel,
permission requests, and the session's own model catalog. The older SDK runtime
has no cancel, so it stays as a fallback.

Why not `dsh web`? It is a long-running browser UI; its `/api` is that UI's
transport, behind a browser-trust fence, and it never exits, so it cannot be a
one-shot target.

## Limitations

- The SDK transport has no per-turn cancel. ACP does: `cancel` stops the running
  turn and the prompt returns `stop_reason: cancelled`.
- ACP binds the working directory when a session is created. To work elsewhere,
  start a new session or resume this one with the new `cwd`.
- Sessions live in the runtime process. Across a Hermes restart, only `resume`
  with a persisted dsh session id brings one back.
- One ACP runtime hosts every session; each session carries its own model and
  effort. The SDK transport needs one runtime per route.
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

The tests use a fake `dsh` executable (`tests/fake_dsh.py`) that speaks the same
headless NDJSON, SDK JSON-RPC, and ACP shapes as the real CLI, so they run
offline without dsh or a DeepSeek key. CI runs them on Ubuntu and macOS against
Python 3.10 and 3.13.

## License

MIT
