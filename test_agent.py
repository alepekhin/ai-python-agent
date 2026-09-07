#!/usr/bin/env python3
import http.server
import json
import os
import pty
import select
import sys
import threading
import time
import typing
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import agent

FAKE_DDG_HTML = """\
<html><body>
<div class="result">
  <h2><a class="result__a" href="https://example.com">Example Result</a></h2>
  <a class="result__snippet" href="https://example.com">This is a fake snippet about the topic.</a>
</div>
<div class="result">
  <h2><a class="result__a" href="https://example2.com">Second Result</a></h2>
  <a class="result__snippet" href="https://example2.com">Another snippet with more info.</a>
</div>
</body></html>
"""


class FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    responses: typing.ClassVar[list[str]] = ["REPLY"]

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        body = json.dumps({"message": {"content": self.responses[0]}}).encode("utf-8")
        if len(self.responses) > 1:
            self.responses.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class FakeDDGHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = FAKE_DDG_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class AgentTTYTest(unittest.TestCase):
    server: http.server.HTTPServer
    ddg_server: http.server.HTTPServer
    thread: threading.Thread
    ddg_thread: threading.Thread
    port: int
    ddg_port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

        cls.ddg_server = http.server.HTTPServer(("127.0.0.1", 0), FakeDDGHandler)
        cls.ddg_thread = threading.Thread(target=cls.ddg_server.serve_forever, daemon=True)
        cls.ddg_thread.start()
        cls.ddg_port = cls.ddg_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.ddg_server.shutdown()
        cls.ddg_server.server_close()

    def _run_agent(self, lines: list[str], responses: list[str] | None = None) -> str:
        master, slave = pty.openpty()
        saved_stdin: int = os.dup(0)
        saved_stdout: int = os.dup(1)
        saved_stderr: int = os.dup(2)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(slave)

        old_url: str = agent.OLLAMA_URL
        old_model: str = agent.MODEL
        old_ddg: str = agent.DDG_URL
        agent.OLLAMA_URL = f"http://127.0.0.1:{self.port}/api/chat"
        agent.DDG_URL = f"http://127.0.0.1:{self.ddg_port}/html/"
        agent.MODEL = "test-model"
        if responses is not None:
            FakeOllamaHandler.responses = list(responses)
        sys.stdout.reconfigure(line_buffering=True)

        output: bytes = b""

        def wait_for(needle: bytes, timeout: float = 15) -> None:
            nonlocal output
            deadline = time.time() + timeout
            while needle not in output:
                if time.time() > deadline:
                    raise AssertionError(
                        f"timed out waiting for {needle!r}; got {output!r}"
                    )
                r, _, _ = select.select([master], [], [], 0.2)
                if master in r:
                    try:
                        chunk = os.read(master, 1024)
                    except OSError:
                        raise AssertionError(
                            f"pty closed while waiting for {needle!r}; got {output!r}"
                        )
                    output += chunk

        try:
            t = threading.Thread(target=agent.main, daemon=True)
            t.start()
            for line in lines:
                wait_for(b"You: ")
                time.sleep(0.1)
                os.write(master, (line + "\n").encode("utf-8"))
            wait_for(b"Bye!")
            t.join(timeout=5)
        finally:
            os.close(master)
            os.dup2(saved_stdin, 0)
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdin)
            os.close(saved_stdout)
            os.close(saved_stderr)
            agent.OLLAMA_URL, agent.MODEL, agent.DDG_URL = old_url, old_model, old_ddg
            FakeOllamaHandler.responses = ["REPLY"]
        return output.decode("utf-8", "replace")

    def test_normal_conversation_then_empty_exit(self) -> None:
        output = self._run_agent(["hello\n", "\n"])
        self.assertIn("You: ", output)
        self.assertIn("Thinking...", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)

    def test_exit_on_empty_prompt(self) -> None:
        output = self._run_agent(["\n"])
        self.assertIn("You: ", output)
        self.assertIn("Bye!", output)
        self.assertNotIn("Thinking...", output)
        self.assertNotIn("Assistant:", output)

    def test_search_triggered(self) -> None:
        output = self._run_agent(
            ["what is the weather?\n", "\n"],
            responses=["[SEARCH: current weather]", "It is sunny today."],
        )
        self.assertIn("Thinking...", output)
        self.assertIn("[search] current weather", output)
        self.assertIn("Assistant: It is sunny today.", output)

    def test_search_results_parsed(self) -> None:
        output = self._run_agent(
            ["search for python\n", "\n"],
            responses=["[SEARCH: python tutorial]", "Here are the results."],
        )
        self.assertIn("[search] python tutorial", output)
        self.assertIn("Assistant: Here are the results.", output)

    def test_search_in_middle_of_reply(self) -> None:
        output = self._run_agent(
            ["tell me about x\n", "\n"],
            responses=[
                "Let me look that up.\n[SEARCH: topic x]",
                "Here is what I found.",
            ],
        )
        self.assertIn("[search] topic x", output)
        self.assertIn("Assistant: Here is what I found.", output)


if __name__ == "__main__":
    unittest.main()
