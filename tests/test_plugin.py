"""Offline tests for the dsh plugin, driven by tests/fake_dsh.py."""
import importlib.util
import json
import os
import stat
import tempfile
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

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_tool(self, **kwargs):
        self.tools.append(kwargs["name"])


def _call(action, **kwargs):
    return json.loads(plugin.handle_dsh_session({"action": action, **kwargs}))


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
        plugin.handle_dsh_session({"action": "shutdown"})
        plugin._SESSIONS.clear()
        plugin._SERVERS.clear()

    def start_session(self, **kwargs):
        result = _call("start", cwd=self.tmp, **kwargs)
        self.assertTrue(result["ok"], result)
        return result["session_id"]


class TestRegistration(DshTestCase):
    def test_registers_both_tools(self):
        self.assertEqual(sorted(self.ctx.tools), ["dsh_agent", "dsh_session"])

    def test_settings_override_route_defaults(self):
        ctx = _Ctx({"provider": "custom-provider", "model": "custom-model",
                    "reasoning_effort": "low"})
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


if __name__ == "__main__":
    unittest.main()
