# hermes-dsh

Hermes plugin that drives [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(`dsh`) agents. Two tools:

- **`dsh_agent`** — one-shot. Spawns `dsh --profile headless --json`, parses the
  NDJSON run events, returns the final answer plus the dsh session id.
- **`dsh_session`** — live control over `dsh --profile sdk`, dsh's newline-delimited
  JSON-RPC runtime (server identity `deepseek-harness-sdk-runtime`). Start a
  session, prompt it, stream its reasoning/tool calls, answer approvals, change
  the working directory, resume a persisted session, shut it down.

## Install

```bash
hermes plugins install haroon0x/hermes-dsh
```

Or from a clone:

```bash
git clone https://github.com/haroon0x/hermes-dsh ~/.hermes/plugins/dsh
hermes plugins enable dsh
```

Requirements:

- `dsh` on `PATH`: `npm install -g @deepseek-ai/dsh`
- a dsh profile that resolves to a provider with a key. The plugin's `dsh_agent`
  uses the `headless` profile, and `dsh_session` needs the `sdk` profile to
  mount the provider adapter you name in `provider`. Both are configured in
  `~/.dsh/profiles/<profile>/cordis.patch.yml`, not by this plugin.

## Configuration

```yaml
plugins:
  entries:
    dsh:
      settings:
        provider: deepseek-official     # default
        model: deepseek-chat            # default
        reasoning_effort: max           # default
```

These are the route `dsh_session` asks the SDK runtime to use. The headless
profile owns its own route; nothing here changes it.

Credentials are never stored or read from files by the plugin. It passes the
environment through to dsh and resolves provider keys with
`agent.secret_scope.get_secret` (`DEEPSEEK_API_KEY`, `OPENCODE_GO_API_KEY`,
`OPENCODE_API_KEY`) so multi-profile gateways do not leak another profile's key.

## dsh_agent

```
dsh_agent(task="run the tests and summarise failures",
          cwd="/abs/project", timeout_s=900)
```

Returns `text`, `session_id`, `tool_calls`, `tool_call_count`, `usage`,
`duration_s`, `exit_code`, `stderr_tail`. Pass the `session_id` back to continue
the same persisted dsh session. A timeout kills the child and returns the
partial answer with `timed_out: true`.

## dsh_session

| action | what it does |
| --- | --- |
| `start` | new session in `cwd` → `session_id` |
| `prompt` | send a turn, wait for `turn/end`, return the answer and tool calls |
| `resume` | reattach a persisted dsh session id after a restart |
| `cwd` | read the session directory, or change it with `path` |
| `list` | live sessions and their status |
| `shutdown` | stop every SDK runtime this plugin started |

A prompt returns the last assistant text, every tool call it made (name plus a
truncated argument string), reasoning character count, event count, turn count,
and approval decisions.

### Approvals

`approval` decides what happens when dsh asks to escalate its sandbox:

- `ask` (default) — refuse the escalation, record it in `pending_approvals`, and
  return guidance. Nothing runs; the model puts the request to the user and can
  retry with `allow` if they agree.
- `allow` — grant `allowed-once` for that operation.
- `reject` — refuse.

Grants never persist beyond the operation, and this plugin never auto-approves.
`ask_user_question` from dsh is answered `{cancelled: true}` because a tool call
has no interactive user; the dismissal is not treated as permission.

## Limitations

- The SDK protocol has no per-turn cancel. A prompt that times out keeps running
  inside dsh; only `shutdown` stops the runtime (and every session in it).
- Sessions live in the runtime process. They survive a Hermes restart only via
  `resume` with a persisted dsh session id.
- Each route (provider/model/effort) gets one runtime; prompts on the same route
  serialize.
- `dsh --profile web` is a long-running browser UI server and is not a valid
  target. The SDK is dsh's automation surface; the web `/api` is the UI's own
  transport behind a browser-trust fence.

## What it runs

The plugin starts `dsh` subprocesses and speaks to them over stdio. dsh runs its
own sandboxed tools; this plugin makes no network calls of its own, writes no
files outside dsh's own state, and stores no credentials. A run can execute
shell commands and edit files in the session's working directory, subject to
dsh's permission preset and the approval rules above.

## Development

```bash
python3 -m unittest discover -s tests -v
hermes plugins validate .
```

The tests use a fake `dsh` executable (`tests/fake_dsh.py`) that emits the same
NDJSON and JSON-RPC shapes as the real CLI, so they run offline without dsh or a
DeepSeek key.

## License

MIT
