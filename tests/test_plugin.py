"""Offline tests for the dsh plugin, driven by tests/fake_dsh.py."""
import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(os.path.dirname(HERE), "__init__.py")
FAKE = os.path.join(HERE, "fake_dsh.py")

spec = importlib.util.spec_from_file_location("dsh_plugin", PLUGIN)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class _Ctx:
    """Minimal PluginContext stand-in: settings + registration."""

    def __init__(self, settings=None):
        self.settings = settings or {}
        self.tools = []
        self.hooks = []

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_tool(self, **kwargs):
        self.tools.append(kwargs["name"])

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))


def _call(action, transport="sdk", **kwargs):
    payload = {"action": action, **kwargs}
    if transport:
        payload["transport"] = transport
    return json.loads(plugin.handle_dsh_session(payload))


class DshTestCase(unittest.TestCase):
    def setUp(self):
        os.chmod(FAKE, os.stat(FAKE).st_mode | stat.S_IXUSR)
        self.addCleanup(self._stop)
        self.tmp = tempfile.mkdtemp(prefix="dsh-test-")
        self.ctx = _Ctx()
        plugin.register(self.ctx)
        self._fake = mock.patch.object(plugin, "_find_dsh", return_value=FAKE)
        self._fake.start()
        self.addCleanup(self._fake.stop)

    def _stop(self):
        plugin._shutdown_all()

    def start_session(self, **kwargs):
        result = _call("start", cwd=self.tmp, **kwargs)
        self.assertTrue(result["ok"], result)
        return result["session_id"]


class TestRegistration(DshTestCase):
    def test_registers_both_tools(self):
        self.assertEqual(sorted(self.ctx.tools), ["dsh_agent", "dsh_session"])

    def test_registers_the_pre_tool_call_hook(self):
        self.assertEqual([name for name, _ in self.ctx.hooks], ["pre_tool_call"])
        self.assertIs(self.ctx.hooks[0][1], plugin._pre_tool_call)

    def test_settings_override_route_defaults(self):
        ctx = _Ctx({"route.provider": "custom-provider", "route.model": "custom-model",
                    "route.reasoning_effort": "low"})
        plugin.register(ctx)
        self.assertEqual(plugin._DEFAULT_PROVIDER, "custom-provider")
        self.assertEqual(plugin._DEFAULT_MODEL, "custom-model")
        self.assertEqual(plugin._DEFAULT_EFFORT, "low")
        # Restore the defaults the other tests rely on.
        plugin.register(_Ctx())


class TestHeadless(DshTestCase):
    def test_returns_final_answer_and_usage(self):
        out = json.loads(plugin.handle_dsh_agent({"task": "say hi", "cwd": self.tmp}))
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["text"], "FAKE_OK")
        self.assertEqual(out["session_id"], "fake-session-1")
        self.assertEqual(out["usage"]["totalTokens"], 12)
        self.assertEqual(out["exit_code"], 0)

    def test_nonzero_exit_without_final_is_an_error(self):
        out = json.loads(plugin.handle_dsh_agent({"task": "FAIL please", "cwd": self.tmp}))
        self.assertFalse(out["ok"])
        self.assertIn("exited 3", out["error"])
        self.assertIn("deliberate failure", out["stderr_tail"] or "")

    def test_session_id_is_echoed_for_resume(self):
        out = json.loads(plugin.handle_dsh_agent(
            {"task": "hi", "cwd": self.tmp, "session_id": "prior-session"}))
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["resumed_from"], "prior-session")

    def test_missing_binary_is_reported(self):
        with mock.patch.object(plugin, "_find_dsh", return_value=None):
            out = json.loads(plugin.handle_dsh_agent({"task": "hi", "cwd": self.tmp}))
        self.assertFalse(out["ok"])
        self.assertIn("not on PATH", out["error"])


class TestSdkSessions(DshTestCase):
    def test_start_prompt_and_shutdown(self):
        sid = self.start_session()
        out = _call("prompt", session_id=sid, task="do the thing")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["text"], "FAKE_OK")
        self.assertEqual(out["status"], "idle")
        self.assertEqual(out["turns"], 1)
        listed = _call("list")
        self.assertEqual(len(listed["sessions"]), 1)
        self.assertEqual(listed["sessions"][0]["session_id"], sid)
        stopped = _call("shutdown")
        self.assertEqual(stopped["runtimes_closed"], 1)

    def test_prompt_requires_a_known_session(self):
        out = _call("prompt", session_id="nope", task="hi")
        self.assertFalse(out["ok"])
        self.assertIn("unknown session", out["error"])

    def test_cwd_read_and_set(self):
        sid = self.start_session()
        self.assertEqual(_call("cwd", session_id=sid)["cwd"], self.tmp)
        moved = _call("cwd", session_id=sid, path="/tmp")
        self.assertEqual(moved["cwd"], "/tmp")

    def test_resume_attaches_existing_session(self):
        out = _call("resume", session_id="persisted-1", cwd=self.tmp)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["session_id"], "persisted-1")

    def test_approval_ask_refuses_and_surfaces_the_request(self):
        sid = self.start_session()
        out = _call("prompt", session_id=sid, task="ESCALATE please")
        self.assertTrue(out["ok"], out)
        self.assertIn("approval=rejected", out["text"])
        self.assertEqual(len(out["pending_approvals"]), 1)
        request = out["pending_approvals"][0]
        self.assertEqual(request["tool"], "write")
        self.assertEqual(request["decision"], "needs-user")
        self.assertIn("approval='allow'", out["guidance"])

    def test_approval_allow_grants_once(self):
        sid = self.start_session()
        out = _call("prompt", session_id=sid, task="ESCALATE please", approval="allow")
        self.assertIn("approval=allowed-once", out["text"])
        self.assertEqual(out["approvals"][0]["outcome"], "allowed-once")
        self.assertEqual(out["pending_approvals"], [])

    def test_approval_reject_never_grants(self):
        sid = self.start_session()
        out = _call("prompt", session_id=sid, task="ESCALATE please", approval="reject")
        self.assertIn("approval=rejected", out["text"])
        self.assertEqual(out["approvals"][0]["decision"], "rejected")

    def test_user_question_is_surfaced_not_answered(self):
        sid = self.start_session()
        out = _call("prompt", session_id=sid, task="QUESTION please")
        self.assertIn("question=cancelled", out["text"])
        self.assertEqual(len(out["pending_questions"]), 1)
        self.assertEqual(out["pending_questions"][0]["answer"], "cancelled")
        self.assertIn("asked the user a question", out["guidance"])

    def test_events_cursor_replays_and_advances(self):
        sid = self.start_session()
        _call("prompt", session_id=sid, task="hello")
        first = _call("events", session_id=sid, since_seq=0)
        self.assertTrue(first["ok"], first)
        self.assertGreater(first["next_seq"], 0)
        self.assertTrue(first["events"])
        second = _call("events", session_id=sid, since_seq=first["next_seq"])
        self.assertEqual(second["events"], [])
        self.assertEqual(second["next_seq"], first["next_seq"])


class TestCleanup(DshTestCase):
    def test_shutdown_all_stops_runtimes_and_forgets_sessions(self):
        sid = self.start_session()
        _call("prompt", session_id=sid, task="hello")
        plugin._shutdown_all()
        self.assertEqual(plugin._SERVERS, {})
        self.assertEqual(plugin._SESSIONS, {})
        gone = _call("events", session_id=sid, since_seq=0)
        self.assertFalse(gone["ok"])


class TestFanout(DshTestCase):
    def test_runs_agents_in_parallel_and_reports_each(self):
        out = _call("fanout", cwd=self.tmp, max_parallel=3,
                    tasks=["one", "two", "three"])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["agents"], 3)
        self.assertEqual(out["failed"], 0)
        self.assertEqual([r["text"] for r in out["results"]], ["FAKE_OK"] * 3)
        self.assertEqual(len({r["session_id"] for r in out["results"]}), 3)
        listed = _call("list")
        self.assertEqual(len(listed["sessions"]), 3)

    def test_accepts_objects_with_per_task_cwd(self):
        out = _call("fanout", tasks=[{"task": "one", "cwd": self.tmp},
                                     {"task": "two", "cwd": self.tmp}])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["agents"], 2)

    def test_empty_task_list_is_rejected(self):
        out = _call("fanout", tasks=[])
        self.assertFalse(out["ok"])
        self.assertIn("tasks is required", out["error"])

    def test_empty_task_text_is_rejected(self):
        out = _call("fanout", tasks=["  "])
        self.assertFalse(out["ok"])
        self.assertIn("empty task", out["error"])

    def test_per_task_route_overrides(self):
        out = _call("fanout", cwd=self.tmp, tasks=[
            {"task": "one", "provider": "deepseek-official", "model": "deepseek-chat"},
            {"task": "two"},
        ])
        self.assertTrue(out["ok"], out)
        routes = {(r["provider"], r["model"]) for r in out["results"]}
        self.assertIn(("deepseek-official", "deepseek-chat"), routes)


class TestAcp(DshTestCase):
    def start_acp(self):
        out = _call("start", transport="acp", cwd=self.tmp)
        self.assertTrue(out["ok"], out)
        return out["session_id"]

    def test_start_applies_route_and_reports_config(self):
        out = _call("start", transport="acp", cwd=self.tmp)
        self.assertTrue(out["ok"], out)
        self.assertIsNone(out["config_errors"])
        self.assertEqual(out["config"]["model"]["current"],
                         '["deepseek-official","deepseek-chat"]')
        self.assertEqual(out["config"]["reasoning_effort"]["current"], "max")

    def test_prompt_returns_text_and_usage(self):
        sid = self.start_acp()
        out = _call("prompt", transport="acp", session_id=sid, task="hello")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["text"], "FAKE_ACP_OK")
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"], {"used": 10, "size": 100})

    def test_approval_ask_refuses_and_surfaces(self):
        sid = self.start_acp()
        out = _call("prompt", transport="acp", session_id=sid, task="ESCALATE")
        self.assertEqual(out["pending_approvals"][0]["decision"], "needs-user")
        self.assertIn("reject-once", out["text"])
        self.assertTrue(out["guidance"])

    def test_approval_allow_grants_once(self):
        sid = self.start_acp()
        out = _call("prompt", transport="acp", session_id=sid, task="ESCALATE",
                    approval="allow")
        self.assertEqual(out["approvals"][0]["option"], "allow_once")
        self.assertIn("allow-once", out["text"])

    def test_cancel_stops_the_turn(self):
        sid = self.start_acp()
        box = {}

        def run():
            box.update(_call("prompt", transport="acp", session_id=sid,
                             task="SLOW", timeout_s=60))

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.5)
        cancelled = _call("cancel", transport="acp", session_id=sid)
        self.assertTrue(cancelled["ok"], cancelled)
        thread.join(30)
        self.assertEqual(box.get("stop_reason"), "cancelled")
        self.assertTrue(box.get("ok"), box)

    def test_events_cursor_advances(self):
        sid = self.start_acp()
        _call("prompt", transport="acp", session_id=sid, task="hello")
        first = _call("events", transport="acp", session_id=sid, since_seq=0)
        self.assertTrue(first["ok"], first)
        self.assertGreater(first["next_seq"], 0)
        self.assertTrue(first["events"])
        second = _call("events", transport="acp", session_id=sid,
                       since_seq=first["next_seq"])
        self.assertEqual(second["events"], [])

    def test_config_set_changes_current_value(self):
        sid = self.start_acp()
        out = _call("config", transport="acp", session_id=sid,
                    config_id="reasoning_effort", value="low")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["config"]["reasoning_effort"]["current"], "low")

    def test_fanout_runs_two_sessions(self):
        out = _call("fanout", transport="acp", cwd=self.tmp, tasks=["one", "two"])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["agents"], 2)
        self.assertEqual([r["text"] for r in out["results"]], ["FAKE_ACP_OK"] * 2)

    def test_cwd_set_is_rejected(self):
        sid = self.start_acp()
        out = _call("cwd", transport="acp", session_id=sid, path=self.tmp)
        self.assertFalse(out["ok"])
        self.assertIn("binds the working directory", out["error"])


class TestApprovalGate(DshTestCase):
    def test_allow_escalates_to_the_human_approval_gate(self):
        directive = plugin._pre_tool_call(
            "dsh_session", {"action": "prompt", "approval": "allow"})
        self.assertEqual(directive["action"], "approve")
        self.assertIn("approval=\"allow\"", directive["message"])
        self.assertIn("prompt", directive["message"])
        self.assertTrue(directive["rule_key"])

    def test_allow_is_normalized_before_gating(self):
        directive = plugin._pre_tool_call(
            "dsh_session", {"action": "fanout", "approval": "  ALLOW  "})
        self.assertEqual(directive["action"], "approve")
        self.assertIn("fanout", directive["message"])

    def test_ask_and_reject_never_escalate(self):
        for approval in ("ask", "reject", "", None):
            args = {"action": "prompt"}
            if approval is not None:
                args["approval"] = approval
            self.assertIsNone(plugin._pre_tool_call("dsh_session", args), approval)

    def test_actions_without_dsh_approvals_are_not_gated(self):
        for action in ("start", "events", "cancel", "config", "list", "shutdown",
                       "resume", "cwd"):
            self.assertIsNone(
                plugin._pre_tool_call("dsh_session", {"action": action,
                                                      "approval": "allow"}), action)

    def test_other_tools_are_not_gated(self):
        self.assertIsNone(plugin._pre_tool_call("terminal", {"approval": "allow"}))
        self.assertIsNone(plugin._pre_tool_call("dsh_agent", {"approval": "allow"}))

    def test_malformed_arguments_are_ignored(self):
        self.assertIsNone(plugin._pre_tool_call("dsh_session", None))
        self.assertIsNone(plugin._pre_tool_call("dsh_session", "not-a-dict"))


class TestChildEnv(DshTestCase):
    def test_allowlists_host_vars_and_drops_unrelated_secrets(self):
        fake = {
            "PATH": "/usr/bin", "HOME": "/home/tester", "SHELL": "/bin/zsh",
            "TERM": "xterm-256color", "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
            "TMPDIR": "/tmp/tester", "NODE_EXTRA_CA_CERTS": "/tmp/ca.pem",
            "SSL_CERT_FILE": "/tmp/cert.pem", "XDG_CONFIG_HOME": "/x/cfg",
            "DSH_HOME": "/x/dsh", "HTTPS_PROXY": "http://proxy:3128",
            "OPENCODE_GO_API_KEY": "go-key",
            "ANTHROPIC_API_KEY": "must-not-leak", "AWS_SECRET_ACCESS_KEY": "must-not-leak",
            "GITHUB_TOKEN": "must-not-leak", "HERMES_OPENAI_KEY": "must-not-leak",
        }
        with mock.patch.dict(os.environ, fake, clear=True), \
                mock.patch.object(plugin, "_secret", return_value=None):
            env = plugin._child_env()
        for name in ("PATH", "HOME", "SHELL", "TERM", "LANG", "LC_ALL", "TMPDIR",
                     "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "XDG_CONFIG_HOME",
                     "DSH_HOME", "HTTPS_PROXY", "OPENCODE_GO_API_KEY"):
            self.assertEqual(env.get(name), fake[name], name)
        for name in ("ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
                     "HERMES_OPENAI_KEY"):
            self.assertNotIn(name, env)

    def test_resolves_one_configured_provider_key_from_the_secret_scope(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True), \
                mock.patch.object(plugin, "_secret",
                                  side_effect=lambda name: {
                                      "OPENCODE_API_KEY": "opencode-key"}.get(name)):
            env = plugin._child_env()
        self.assertEqual(env["OPENCODE_API_KEY"], "opencode-key")
        self.assertNotIn("OPENCODE_GO_API_KEY", env)
        self.assertNotIn("DEEPSEEK_API_KEY", env)

    def test_parent_environment_wins_over_the_secret_scope(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin",
                                          "DEEPSEEK_API_KEY": "from-shell"}, clear=True), \
                mock.patch.object(plugin, "_secret", return_value="from-scope"):
            env = plugin._child_env()
        self.assertEqual(env["DEEPSEEK_API_KEY"], "from-shell")

    def test_binary_directory_is_prepended_to_path(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True), \
                mock.patch.object(plugin, "_secret", return_value=None):
            env = plugin._child_env("/opt/node/bin/dsh")
        self.assertEqual(env["PATH"], "/opt/node/bin" + os.pathsep + "/usr/bin")
        # Already-present directories are not duplicated.
        with mock.patch.dict(os.environ, {"PATH": "/opt/node/bin:/usr/bin"}, clear=True), \
                mock.patch.object(plugin, "_secret", return_value=None):
            self.assertEqual(plugin._child_env("/opt/node/bin/dsh")["PATH"],
                             "/opt/node/bin:/usr/bin")

    def test_spawned_dsh_gets_the_filtered_environment(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(plugin, "_child_env", return_value={"PATH": "/x"}):
            out = json.loads(plugin.handle_dsh_agent({"task": "hi", "cwd": self.tmp}))
        self.assertTrue(out["ok"], out)
        self.assertEqual(captured["env"], {"PATH": "/x"})


if __name__ == "__main__":
    unittest.main()
