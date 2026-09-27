#!/usr/bin/env python3
import contextlib
import http.server
import json
import os
import pty
import select
import sys
import tempfile
import threading
import time
import types
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
    requests: typing.ClassVar[list[list[dict[str, str]]]] = []

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        FakeOllamaHandler.requests.append(json.loads(raw.decode("utf-8"))["messages"])
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


class BrokenModule(types.ModuleType):
    """Module that raises ImportError on any attribute access, as if missing."""

    def __getattr__(self, name: str) -> typing.Any:
        raise ImportError(f"No module named {self.__name__!r}")


class FakeWhisperSegment:
    def __init__(self, text: str) -> None:
        self.text = text


@contextlib.contextmanager
def fake_voice_modules(
    text: str, broken: tuple[str, ...] = (), mic_error: bool = False
) -> typing.Iterator[dict[str, typing.Any]]:
    """Install fake sounddevice/faster_whisper modules for the duration.

    Args:
        text: transcription the fake Whisper model should return.
        broken: module names to fake as not installed.
        mic_error: make opening the microphone fail.

    Yields:
        A dict recording the fake model name and the transcribed audio.
    """
    import numpy as np

    calls: dict[str, typing.Any] = {"model": None, "transcribed": []}

    class PortAudioError(Exception):
        pass

    class RawInputStream:
        def __init__(self, **kwargs: typing.Any) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> typing.Self:
            if mic_error:
                raise PortAudioError("no default input device")
            blocksize = self.kwargs["blocksize"]
            self.kwargs["callback"](
                np.zeros((blocksize, 1), dtype="float32"), blocksize, None, None
            )
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    class WhisperModel:
        def __init__(self, name: str, **kwargs: typing.Any) -> None:
            calls["model"] = name

        def transcribe(
            self, audio: typing.Any, **kwargs: typing.Any
        ) -> tuple[list[typing.Any], object]:
            calls["transcribed"].append((len(audio), kwargs))
            return [FakeWhisperSegment(text)], object()

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.RawInputStream = RawInputStream  # type: ignore[attr-defined]
    sounddevice.PortAudioError = PortAudioError  # type: ignore[attr-defined]
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = WhisperModel  # type: ignore[attr-defined]

    fakes = {"sounddevice": sounddevice, "faster_whisper": faster_whisper}
    saved = {name: sys.modules.get(name) for name in fakes}
    for name, fake in fakes.items():
        sys.modules[name] = BrokenModule(name) if name in broken else fake

    agent.load_whisper.cache_clear()
    try:
        yield calls
    finally:
        agent.load_whisper.cache_clear()
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


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

    def setUp(self) -> None:
        # Point the persisted history at a throwaway file, so tests never read
        # or write the real one in the user's home directory.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        agent.HISTORY_FILE = str(Path(self.tmpdir.name) / "state" / "history")
        if agent.readline is not None:
            agent.readline.clear_history()
            self.addCleanup(agent.readline.clear_history)

    def _run_agent(
        self,
        lines: list[str],
        responses: list[str] | None = None,
        markers: list[bytes | None] | None = None,
    ) -> str:
        """Run the agent on a pty, feeding it `lines` one by one.

        Args:
            lines: text written to the pty before each expected marker.
            responses: canned Ollama replies, one per request.
            markers: output to wait for before writing each line, defaults
                     to the "You: " prompt for every line. None writes the
                     line right away, for input typed ahead of the prompt.
        """
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
        FakeOllamaHandler.requests = []
        sys.stdout.reconfigure(line_buffering=True)

        output: bytes = b""
        searched: int = 0  # only look at output produced after the last match

        def wait_for(needle: bytes, timeout: float = 15) -> None:
            nonlocal output, searched
            deadline = time.time() + timeout
            while needle not in output[searched:]:
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
            searched = output.index(needle, searched) + len(needle)

        try:
            t = threading.Thread(target=agent.main, daemon=True)
            t.start()
            for i, line in enumerate(lines):
                marker = markers[i] if markers is not None else b"You: "
                if marker is not None:
                    wait_for(marker)
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

    def _last_user_prompts(self) -> list[str]:
        return [
            m["content"] for m in FakeOllamaHandler.requests[-1] if m["role"] == "user"
        ]

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_up_arrow_repeats_previous_command(self) -> None:
        output = self._run_agent(["hello", "\x1b[A", ""], responses=["ONE", "TWO"])
        self.assertIn("Assistant: ONE", output)
        self.assertIn("Assistant: TWO", output)
        self.assertIn("Bye!", output)
        self.assertEqual(len(FakeOllamaHandler.requests), 2)
        self.assertEqual(self._last_user_prompts(), ["hello", "hello"])

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_down_arrow_returns_to_empty_input(self) -> None:
        # Walking up and back down leaves the input empty, so the agent exits
        # and never shows a third prompt: the last line is typed ahead.
        output = self._run_agent(
            ["hello", "\x1b[A\x1b[B", ""],
            responses=["ONE", "TWO"],
            markers=[b"You: ", b"Assistant: ONE", None],
        )
        self.assertIn("Assistant: ONE", output)
        self.assertIn("Bye!", output)
        self.assertEqual(len(FakeOllamaHandler.requests), 1)
        self.assertNotIn("Assistant: TWO", output)

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_prompts_persisted_for_next_session(self) -> None:
        self._run_agent(["hello", ""])
        history = Path(agent.HISTORY_FILE).read_text(encoding="utf-8")
        self.assertIn("hello", history)

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_up_arrow_recalls_prompt_from_previous_session(self) -> None:
        Path(agent.HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(agent.HISTORY_FILE).write_text("prompt from last time\n", encoding="utf-8")
        output = self._run_agent(["\x1b[A", ""], responses=["REPLY"])
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        self.assertEqual(self._last_user_prompts(), ["prompt from last time"])

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_history_file_capped_to_limit(self) -> None:
        Path(agent.HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        stored = "".join(f"old prompt {i}\n" for i in range(agent.HISTORY_LIMIT + 5))
        Path(agent.HISTORY_FILE).write_text(stored, encoding="utf-8")
        agent.load_input_history()
        length = agent.readline.get_current_history_length()
        history = [agent.readline.get_history_item(i + 1) for i in range(length)]
        self.assertEqual(len(history), agent.HISTORY_LIMIT)
        self.assertEqual(history[-1], f"old prompt {agent.HISTORY_LIMIT + 4}")

    def test_voice_prompt_is_transcribed_and_sent(self) -> None:
        with fake_voice_modules("tell me a joke") as calls:
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        self.assertIn("Recording... press Enter to stop", output)
        self.assertIn("You (voice): tell me a joke", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        self.assertEqual(calls["model"], agent.WHISPER_MODEL)
        self.assertEqual(self._last_user_prompts(), ["tell me a joke"])

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_voice_prompt_kept_in_input_history(self) -> None:
        with fake_voice_modules("remember me"):
            self._run_agent(
                [agent.VOICE_COMMAND, "", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        history = Path(agent.HISTORY_FILE).read_text(encoding="utf-8")
        self.assertIn("remember me", history)

    def test_voice_prompt_without_packages_reports_error(self) -> None:
        with fake_voice_modules("", broken=("sounddevice", "faster_whisper")):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("pip install faster-whisper sounddevice", output)
        self.assertIn("Assistant: REPLY", output)
        # The failed voice prompt is not sent, the dialog keeps working.
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_voice_prompt_with_broken_microphone_reports_error(self) -> None:
        with fake_voice_modules("", mic_error=True):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("no default input device", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_silent_voice_prompt_is_ignored(self) -> None:
        with fake_voice_modules(""):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", "hello", ""],
                markers=[b"You: ", b"Recording...", b"You: ", b"You: "],
            )
        self.assertIn("Nothing recognized", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(self._last_user_prompts(), ["hello"])


if __name__ == "__main__":
    unittest.main()
