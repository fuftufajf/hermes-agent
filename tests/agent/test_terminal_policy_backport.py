"""Desktop batches publish consent together, never shell effects together."""

import json
import queue
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent
from gateway.session_context import clear_session_vars
from tools import approval
from tools.thread_context import propagate_context_to_thread


def _call(call_id, command):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name="terminal", arguments=json.dumps({"command": command})))


def _agent():
    from tools.terminal_tool import TERMINAL_SCHEMA
    from tools.file_tools import READ_FILE_SCHEMA
    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": schema}
            for schema in (TERMINAL_SCHEMA, READ_FILE_SCHEMA)
        ]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                       quiet_mode=True, skip_context_files=True, skip_memory=True, platform="desktop")




def test_switching_yolo_off_mid_batch_re_gates_later_commands(tmp_path, monkeypatch):
    """A prepared approval nobody answered is policy: the policy in force at execution governs."""
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
    from tools.terminal_tool_lifecycle import cleanup_vm

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")
    monkeypatch.setattr("tools.approval._tirith_scan", lambda command: {"action": "allow"})
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    key = "yolo-off-terminal-batch"
    agent = _agent()
    agent._flush_messages_to_session_db = lambda *a, **kw: True
    published = queue.Queue()
    approval.register_gateway_notify(key, published.put)
    from tui_gateway import server
    monkeypatch.setattr(server, "_sessions", {key: {
        "session_key": key, "source": "desktop", "agent": agent, "cwd": str(tmp_path)}})
    tokens = server._set_session_context(key)
    calls = [_call(f"c{i}", f"rm -rf absent-{i}; touch ran-{i}") for i in range(3)]
    # The user switches YOLO off once the first command has run.
    agent.tool_start_callback = lambda call_id, name, args: call_id == "c1" and approval.disable_session_yolo(key)
    messages, errors = [], []
    approval.enable_session_yolo(key)

    def run():
        try:
            agent._execute_tool_calls(SimpleNamespace(tool_calls=calls), messages, key)
        except BaseException as exc:
            errors.append(exc)

    with ExitStack() as scope:
        scope.callback(reset_terminal_scope, set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)}))
        worker = threading.Thread(target=propagate_context_to_thread(run), daemon=True)
        worker.start()
        try:
            for _ in range(2):
                request = published.get(timeout=15)
                assert approval.resolve_gateway_approval(key, "deny", request_id=request["request_id"]) == 1
            worker.join(timeout=15)
            assert not worker.is_alive() and errors == []
            assert sorted(p.name for p in tmp_path.glob("ran-*")) == ["ran-0"]
        finally:
            agent.interrupt("test cleanup")
            approval.unregister_gateway_notify(key)
            approval.clear_session(key)
            worker.join(timeout=5)
            cleanup_vm(key)
            clear_session_vars(tokens)
