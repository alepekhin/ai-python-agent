#!/usr/bin/env python3
import http.server
import json
import os
import pty
import select
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import agent


class FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"message": {"content": "REPLY"}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class AgentTTYTest(unittest.TestCase):
    server: http.server.HTTPServer
    thread: threading.Thread
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _run_agent(self, lines: list[str]) -> str:
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
        agent.OLLAMA_URL = f"http://127.0.0.1:{self.port}/api/chat"
        agent.MODEL = "test-model"
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
            agent.OLLAMA_URL, agent.MODEL = old_url, old_model
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


if __name__ == "__main__":
    unittest.main()
