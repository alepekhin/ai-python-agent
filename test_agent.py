#!/usr/bin/env python3
import base64
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
    payloads: typing.ClassVar[list[dict[str, typing.Any]]] = []
    # When set, a request carrying an audio block is rejected with this reason.
    audio_error: typing.ClassVar[str | None] = None

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8"))
        FakeOllamaHandler.payloads.append(payload)
        if self.audio_error and self._has_audio(payload):
            self._send(500, json.dumps({"error": self.audio_error}).encode("utf-8"))
            return
        body = json.dumps(
            {"choices": [{"message": {"content": self.responses[0]}}]}
        ).encode("utf-8")
        if len(self.responses) > 1:
            self.responses.pop(0)
        self._send(200, body)

    @staticmethod
    def _has_audio(payload: dict[str, typing.Any]) -> bool:
        return any(
            isinstance(message["content"], list)
            and any(part["type"] == "input_audio" for part in message["content"])
            for message in payload["messages"]
        )

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
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


class FakePiperChunk:
    def __init__(self, sample_rate: int = 22050) -> None:
        import numpy as np

        self.sample_rate = sample_rate
        self.audio_float_array = np.zeros(1600, dtype="float32")


@contextlib.contextmanager
def fake_voice_modules(
    broken: tuple[str, ...] = (),
    mic_error: bool = False,
    tts_error: bool = False,
    levels: list[float] | None = None,
    interrupt_wait: bool = False,
) -> typing.Iterator[dict[str, typing.Any]]:
    """Install fake sounddevice/piper modules for the duration.

    Args:
        broken: module names to fake as not installed.
        mic_error: make opening the microphone fail.
        tts_error: make the fake Piper voice fail to load.
        levels: RMS level of each block the fake microphone delivers; by
                default a quiet room and a word in it.
        interrupt_wait: make waiting for playback raise KeyboardInterrupt,
                as Ctrl+C does while a reply is being read out loud.

    Yields:
        A dict recording the voice files requested/downloaded, the text
        handed to the synthesizer and the chunks played.
    """
    import numpy as np

    calls: dict[str, typing.Any] = {
        "downloaded": [],
        "loaded": [],
        "spoken": [],
        "played": [],
    }

    class PortAudioError(Exception):
        pass

    class RawInputStream:
        def __init__(self, **kwargs: typing.Any) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> typing.Self:
            if mic_error:
                raise PortAudioError("no default input device")
            blocksize = self.kwargs["blocksize"]
            spoken = (
                [0.05] * agent.VOICE_NOISE_BLOCKS + [0.2, 0.2]
                if levels is None
                else levels
            )
            for level in spoken:
                self.kwargs["callback"](
                    np.full((blocksize, 1), level, dtype="float32"),
                    blocksize,
                    None,
                    None,
                )
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def play(data: typing.Any, samplerate: typing.Any) -> None:
        calls["played"].append((len(data), samplerate))

    def wait() -> None:
        if interrupt_wait:
            raise KeyboardInterrupt

    def stop() -> None:
        pass

    class PiperVoice:
        def __init__(self, name: str) -> None:
            self.name = name

        @staticmethod
        def load(
            model_path: typing.Any, config_path: typing.Any = None, **kwargs: typing.Any
        ) -> typing.Self:
            if tts_error:
                raise RuntimeError("broken onnx model")
            calls["loaded"].append((str(model_path), str(config_path)))
            return PiperVoice(Path(str(model_path)).stem)

        def synthesize(self, text: str) -> list[typing.Any]:
            calls["spoken"].append((self.name, text))
            return [FakePiperChunk()]

    def download_voice(voice: str, download_dir: typing.Any) -> None:
        calls["downloaded"].append((voice, str(download_dir)))
        (Path(download_dir) / f"{voice}.onnx").write_bytes(b"")
        (Path(download_dir) / f"{voice}.onnx.json").write_text("{}")

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.RawInputStream = RawInputStream  # type: ignore[attr-defined]
    sounddevice.PortAudioError = PortAudioError  # type: ignore[attr-defined]
    sounddevice.play = play  # type: ignore[attr-defined]
    sounddevice.wait = wait  # type: ignore[attr-defined]
    sounddevice.stop = stop  # type: ignore[attr-defined]
    download_voices = types.ModuleType("piper.download_voices")
    download_voices.download_voice = download_voice  # type: ignore[attr-defined]
    piper = types.ModuleType("piper")
    piper.PiperVoice = PiperVoice  # type: ignore[attr-defined]
    piper.download_voices = download_voices  # type: ignore[attr-defined]

    fakes = {
        "sounddevice": sounddevice,
        "piper": piper,
        "piper.download_voices": download_voices,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    for name, fake in fakes.items():
        sys.modules[name] = BrokenModule(name) if name in broken else fake

    agent.load_piper.cache_clear()
    try:
        yield calls
    finally:
        agent.load_piper.cache_clear()
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
        cls.ddg_thread = threading.Thread(
            target=cls.ddg_server.serve_forever, daemon=True
        )
        cls.ddg_thread.start()
        cls.ddg_port = cls.ddg_server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.ddg_server.shutdown()
        cls.ddg_server.server_close()

    def setUp(self) -> None:
        # Point the persisted history and the voice cache at a throwaway
        # directory, so tests never read or write the real ones in the user's
        # home directory.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        agent.HISTORY_FILE = str(Path(self.tmpdir.name) / "state" / "history")
        agent.LAST_PROMPT_FILE = str(
            Path(self.tmpdir.name) / "state" / "last_prompt.json"
        )
        self.addCleanup(setattr, agent, "LAST_PROMPT_FILE", agent.LAST_PROMPT_FILE)
        agent.TTS_DIR = str(Path(self.tmpdir.name) / "state" / "voices")
        # Spoken replies are off unless a test turns them on, so a run that
        # does not fake the voice modules never touches real audio.
        agent.SPEAK_DEFAULT = False
        self.addCleanup(setattr, agent, "SPEAK_DEFAULT", True)
        # Read the files the test creates, not the ones in the real project.
        agent.READ_ROOTS = [self.tmpdir.name]
        self.addCleanup(setattr, agent, "READ_ROOTS", None)
        agent.load_piper.cache_clear()
        self.addCleanup(agent.load_piper.cache_clear)
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
        agent.OLLAMA_URL = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        agent.DDG_URL = f"http://127.0.0.1:{self.ddg_port}/html/"
        agent.MODEL = "test-model"
        if responses is not None:
            FakeOllamaHandler.responses = list(responses)
        FakeOllamaHandler.payloads = []
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

    def test_read_triggered(self) -> None:
        (Path(self.tmpdir.name) / "notes.txt").write_text("hello\nworld\n", "utf-8")
        output = self._run_agent(
            ["what is in notes.txt?\n", "\n"],
            responses=["[READ: notes.txt]", "It says hello and world."],
        )
        self.assertIn("[read] notes.txt", output)
        self.assertIn("Assistant: It says hello and world.", output)
        # The file itself reached the model, as numbered lines.
        self.assertEqual(
            self._last_user_prompts(),
            [
                "what is in notes.txt?",
                "Contents of notes.txt (2 lines):\n1: hello\n2: world\n\n"
                + agent.ANSWER_FROM_RESULT,
            ],
        )

    def test_tag_quoted_in_an_answer_is_not_a_tool_call(self) -> None:
        # A model describing the syntax is not asking for a tool.
        output = self._run_agent(
            ["how do you read files?\n", "\n"],
            responses=["You write [READ: path] on its own line."],
        )
        self.assertIn("Assistant: You write [READ: path] on its own line.", output)
        self.assertEqual(len(FakeOllamaHandler.payloads), 1)

    def test_read_of_missing_file_is_reported_to_the_model(self) -> None:
        output = self._run_agent(
            ["read notes.txt\n", "\n"],
            responses=["[READ: notes.txt]", "There is no such file."],
        )
        self.assertIn("[read] notes.txt", output)
        self.assertIn("Cannot read notes.txt", self._last_user_prompts()[-1])
        self.assertIn("Assistant: There is no such file.", output)

    def test_read_of_binary_file_gives_the_path_only(self) -> None:
        picture = Path(self.tmpdir.name) / "blood.jpg"
        picture.write_bytes(b"\xff\xd8\xff\xe0JPEGBYTES\x00data")
        output = self._run_agent(
            ["what is in blood.jpg?\n", "\n"],
            responses=["[READ: blood.jpg]", "It is a photo, I cannot show it."],
        )
        self.assertIn("[read] blood.jpg", output)
        sent = self._last_user_prompts()[-1]
        self.assertIn("blood.jpg", sent)
        self.assertIn("binary file", sent)
        # No byte of the file itself reached the model.
        self.assertNotIn("JPEGBYTES", sent)
        self.assertIn("Assistant: It is a photo, I cannot show it.", output)

    def test_prompt_naming_a_binary_file_carries_its_path_to_the_model(self) -> None:
        picture = Path(self.tmpdir.name) / "blood.jpg"
        picture.write_bytes(b"\xff\xd8\xff\xe0JPEGBYTES\x00data")
        (Path(self.tmpdir.name) / "notes.txt").write_text("hello\n", "utf-8")
        output = self._run_agent(
            ["compare blood.jpg and notes.txt\n", "\n"], responses=["REPLY"]
        )
        sent = self._last_user_prompts()[0]
        self.assertIn("binary", sent)
        self.assertIn("blood.jpg", sent)
        # A text file is not named in the note: the model can read it anyway.
        self.assertNotIn("notes.txt", sent.split("\n", 1)[1])
        # No byte of the binary file itself reaches the model.
        self.assertNotIn("JPEGBYTES", sent)
        self.assertIn("You: compare blood.jpg and notes.txt", output)

    def test_prompt_without_a_binary_file_is_sent_unchanged(self) -> None:
        (Path(self.tmpdir.name) / "notes.txt").write_text("hello\n", "utf-8")
        self._run_agent(["what is in notes.txt?\n", "\n"], responses=["REPLY"])
        self.assertEqual(self._last_user_prompts()[0], "what is in notes.txt?")

    def test_read_stays_inside_the_allowed_directories(self) -> None:
        secret = Path(self.tmpdir.name).parent / "secret.txt"
        secret.write_text("top secret\n", encoding="utf-8")
        self.addCleanup(secret.unlink)
        output = self._run_agent(
            ["read the secret\n", "\n"],
            responses=[f"[READ: {secret}]", "I cannot read that file."],
        )
        self.assertIn("outside", self._last_user_prompts()[-1])
        self.assertNotIn("top secret", output)

    def test_search_and_read_can_follow_each_other(self) -> None:
        (Path(self.tmpdir.name) / "a.txt").write_text("alpha\n", encoding="utf-8")
        output = self._run_agent(
            ["check a.txt\n", "\n"],
            responses=["[READ: a.txt]", "[SEARCH: a.txt meaning]", "Alpha."],
        )
        self.assertIn("[read] a.txt", output)
        self.assertIn("[search] a.txt meaning", output)
        self.assertIn("Assistant: Alpha.", output)

    def test_tool_is_not_run_forever(self) -> None:
        (Path(self.tmpdir.name) / "a.txt").write_text("alpha\n", encoding="utf-8")
        output = self._run_agent(
            ["check a.txt\n", "\n"],
            responses=["[READ: a.txt]"] * (agent.MAX_TOOL_ROUNDS + 1),
        )
        # One reply to start with, plus one per round the agent is willing to run.
        self.assertEqual(len(FakeOllamaHandler.payloads), agent.MAX_TOOL_ROUNDS + 1)
        self.assertIn("[read] a.txt", output)
        self.assertIn("Bye!", output)

    def test_image_attached_to_the_request(self) -> None:
        picture = Path(self.tmpdir.name) / "blood.jpg"
        picture.write_bytes(b"\xff\xd8\xff\xe0JPEGBYTES\x00data")
        output = self._run_agent(
            ["что на картинке blood.jpg\n", "\n"],
            responses=["[IMAGE: blood.jpg]", "Это анализ крови."],
        )
        self.assertIn("[image] blood.jpg", output)
        self.assertIn("Assistant: Это анализ крови.", output)
        # The second request carries the picture as a base64 data URL.
        urls = self._image_urls(FakeOllamaHandler.payloads[-1])
        self.assertEqual(len(urls), 1)
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))
        self.assertIn(base64.b64encode(picture.read_bytes()).decode(), urls[0])
        texts = self._part_texts(FakeOllamaHandler.payloads[-1])
        self.assertIn("blood.jpg", texts)
        self.assertIn(agent.ANSWER_FROM_RESULT, texts)

    def test_image_of_a_file_that_is_not_one_is_refused(self) -> None:
        (Path(self.tmpdir.name) / "notes.txt").write_text("hello\n", "utf-8")
        self._run_agent(
            ["look at notes.txt\n", "\n"],
            responses=["[IMAGE: notes.txt]", "It is a text file."],
        )
        self.assertIn("not an image format", self._last_user_prompts()[-1])
        self.assertEqual(self._image_urls(FakeOllamaHandler.payloads[-1]), [])

    def test_image_outside_the_allowed_directories_is_refused(self) -> None:
        secret = Path(self.tmpdir.name).parent / "secret.png"
        secret.write_bytes(b"\x89PNG\r\n\x1a\n\x00data")
        self.addCleanup(secret.unlink)
        self._run_agent(
            ["look at the secret\n", "\n"],
            responses=[f"[IMAGE: {secret}]", "I cannot see it."],
        )
        self.assertIn("outside", self._last_user_prompts()[-1])

    def test_image_bigger_than_the_cap_is_not_attached(self) -> None:
        (Path(self.tmpdir.name) / "big.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"x" * 99
        )
        old = agent.IMAGE_MAX_BYTES
        agent.IMAGE_MAX_BYTES = 10
        self.addCleanup(setattr, agent, "IMAGE_MAX_BYTES", old)
        self._run_agent(
            ["look at big.png\n", "\n"], responses=["[IMAGE: big.png]", "No."]
        )
        self.assertIn("at most 10 are attached", self._last_user_prompts()[-1])
        self.assertEqual(self._image_urls(FakeOllamaHandler.payloads[-1]), [])

    def test_older_image_is_dropped_from_the_history(self) -> None:
        for name in ("one.jpg", "two.jpg"):
            (Path(self.tmpdir.name) / name).write_bytes(
                b"\xff\xd8\xff\xe0" + name.encode()
            )
        self._run_agent(
            ["one", "two", "three", ""],
            responses=[
                "[IMAGE: one.jpg]",
                "One.",
                "[IMAGE: two.jpg]",
                "Two.",
                "Three.",
            ],
        )
        # The third turn carries the newest image, and a note where the older
        # one was: its bytes are not sent again.
        last = FakeOllamaHandler.payloads[-1]
        self.assertEqual(len(self._image_urls(last)), 1)
        parts = self._part_texts(last)
        self.assertIn("two.jpg", parts)
        self.assertEqual(parts.count(agent.IMAGE_DROPPED), 1)
        self.assertIn("one.jpg", parts)

    def test_saved_prompt_holds_no_image_data(self) -> None:
        (Path(self.tmpdir.name) / "blood.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"x" * 40
        )
        self._run_agent(
            ["look at blood.jpg\n", "\n"],
            responses=["[IMAGE: blood.jpg]", "It shows a blood test."],
        )
        saved = Path(agent.LAST_PROMPT_FILE).read_text(encoding="utf-8")
        self.assertIn(agent.MEDIA_LOGGED, saved)
        self.assertNotIn(
            base64.b64encode(b"\xff\xd8\xff\xe0" + b"x" * 40).decode(), saved
        )

    def _last_user_prompts(self) -> list[str]:
        return [
            m["content"]
            for m in FakeOllamaHandler.payloads[-1]["messages"]
            if m["role"] == "user" and isinstance(m["content"], str)
        ]

    def _image_urls(self, payload: dict[str, typing.Any]) -> list[str]:
        """URL of every `image_url` part the model was sent in this request."""
        return [
            part["image_url"]["url"]
            for message in payload["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]

    def _part_texts(self, payload: dict[str, typing.Any]) -> str:
        """Every text part of a request, plus its plain text messages."""
        return "\n".join(
            part["text"]
            for message in payload["messages"]
            for part in (
                [{"type": "text", "text": message["content"]}]
                if isinstance(message["content"], str)
                else message["content"]
            )
            if part["type"] == "text"
        )

    def _audio_blocks(self) -> list[dict[str, typing.Any]]:
        """Audio blocks of every `input_audio` part the model was sent."""
        return [
            part["input_audio"]
            for payload in FakeOllamaHandler.payloads
            for message in payload["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "input_audio"
        ]

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_up_arrow_repeats_previous_command(self) -> None:
        output = self._run_agent(["hello", "\x1b[A", ""], responses=["ONE", "TWO"])
        self.assertIn("Assistant: ONE", output)
        self.assertIn("Assistant: TWO", output)
        self.assertIn("Bye!", output)
        self.assertEqual(len(FakeOllamaHandler.payloads), 2)
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
        self.assertEqual(len(FakeOllamaHandler.payloads), 1)
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

    def test_last_prompt_sent_to_the_model_is_saved(self) -> None:
        self._run_agent(["привет", ""], responses=["REPLY"])
        saved = json.loads(Path(agent.LAST_PROMPT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "test-model")
        self.assertEqual(saved["messages"][0]["role"], "system")
        self.assertEqual(saved["messages"][-1]["content"], "привет")
        # The file holds the request, written unescaped so it stays readable.
        self.assertIn("привет", Path(agent.LAST_PROMPT_FILE).read_text("utf-8"))

    def test_last_prompt_file_holds_the_turn_that_ran_a_tool(self) -> None:
        (Path(self.tmpdir.name) / "a.txt").write_text("alpha\n", "utf-8")
        self._run_agent(["check a.txt", ""], responses=["[READ: a.txt]", "Alpha."])
        saved = json.loads(Path(agent.LAST_PROMPT_FILE).read_text(encoding="utf-8"))
        contents = [m["content"] for m in saved["messages"]]
        self.assertIn("Contents of a.txt (1 lines):\n1: alpha", contents[-1])

    def test_voice_prompt_is_sent_as_audio_and_transcribed_by_the_model(
        self,
    ) -> None:
        with fake_voice_modules():
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", ""],
                responses=["tell me a joke", "REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        self.assertIn("Recording... press Enter to stop", output)
        self.assertIn("transcribing with the model", output)
        self.assertIn("You (voice): tell me a joke", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        # The recording reached the model as a WAV, the transcript as text.
        blocks = self._audio_blocks()
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["format"], "wav")
        self.assertEqual(len(base64.b64decode(blocks[0]["data"])) % 2, 0)
        self.assertEqual(self._last_user_prompts(), ["tell me a joke"])

    def test_voice_prompt_ends_on_a_pause_without_enter(self) -> None:
        import io as fake_io
        import wave as fake_wave

        # A word after the room has been measured, then a noisy room for the
        # length of a pause and no keypress: the pause ends the recording,
        # and the lines after it are typed ahead of the prompt.
        room = agent.VOICE_NOISE_BLOCKS
        levels = [0.05] * room + [0.2] + [0.05] * (
            agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS
        )
        with fake_voice_modules(levels=levels):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "hello", ""],
                responses=["dictated", "REPLY"],
                markers=[b"You: ", b"You (voice): dictated", None, None],
            )
        self.assertIn("Heard a pause, ending the prompt.", output)
        self.assertIn("You (voice): dictated", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        self.assertEqual(self._last_user_prompts(), ["dictated", "hello"])
        # Only what was said reached the model: the noise of the pause did not.
        audio = base64.b64decode(self._audio_blocks()[0]["data"])
        with fake_wave.open(fake_io.BytesIO(audio)) as wav:
            self.assertEqual(
                wav.getnframes(),
                (room + 1) * agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000,
            )
            pcm = wav.readframes(wav.getnframes())
        self.assertGreater(max(pcm), 0)  # not the noise that was dropped

    def test_the_opening_bang_is_not_a_word(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        # A microphone that comes up with a bang in its very first block:
        # the opening blocks only measure the room, or that bang would pass
        # for a word and raise the bar above every word after it.
        for i in range(agent.VOICE_NOISE_BLOCKS):
            watcher.add(1.0 if i == 0 else 0.01, block)
        self.assertFalse(watcher.spoken)
        # The bar falls back to the room, so a word after the bang is heard.
        for _ in range(4 * agent.VOICE_NOISE_BLOCKS):
            watcher.add(0.01, block)
        self.assertLess(watcher.threshold, 0.2)
        watcher.add(0.2, block)  # a word over the quiet room
        self.assertTrue(watcher.spoken)
        self.assertEqual(watcher.voice_blocks, 5 * agent.VOICE_NOISE_BLOCKS + 1)

    def test_nothing_spoken_is_not_sent_to_the_model(self) -> None:
        room = agent.VOICE_NOISE_BLOCKS
        with fake_voice_modules(levels=[0.05] * (room * 3)):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", "hello", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"Recording...", b"You: ", b"You: "],
            )
        self.assertIn("Nothing recorded.", output)
        self.assertIn("is what counts as speech", output)
        # The model is asked nothing about the room: it is not a prompt.
        self.assertEqual(self._audio_blocks(), [])
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_typing_the_next_prompt_does_not_stall_the_recording(self) -> None:
        # The whole next prompt, typed while the recording is still running:
        # the Enter at its end stops the recording, and the text belongs to
        # the recording rather than to the prompt that comes after it.
        with fake_voice_modules():
            output = self._run_agent(
                [agent.VOICE_COMMAND, "Capital of France", ""],
                responses=["Paris", "REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        self.assertIn("You (voice): Paris", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        self.assertEqual(self._last_user_prompts(), ["Paris"])

    def test_pause_ends_the_recording_only_after_a_word(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        # A quiet room says nothing, so it must not cut off a recording that
        # has not started yet.
        quiet = agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS + 1
        for _ in range(quiet):
            watcher.add(0.0, block)
        self.assertFalse(watcher.spoken)
        self.assertFalse(watcher.paused)
        # A word, a short breath and the word again: the pause never grows
        # long enough to end anything.
        for level in [0.5, 0.0, 0.0, 0.5]:
            watcher.add(level, block)
        self.assertTrue(watcher.spoken)
        self.assertFalse(watcher.paused)
        # The pause that follows the last word ends the recording, and only
        # the blocks up to that word are worth keeping.
        for _ in range(agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS):
            watcher.add(0.0, block)
        self.assertTrue(watcher.paused)
        self.assertEqual(watcher.voice_blocks, quiet + 4)

    def test_silence_before_the_first_word_is_not_counted(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        # Ten pauses worth of quiet before anybody speaks: none of it counts.
        for _ in range(10 * (agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS)):
            watcher.add(0.0, block)
        self.assertEqual(watcher.silent_ms, 0)
        watcher.add(0.5, block)  # the word
        watcher.add(0.0, block)
        self.assertEqual(watcher.silent_ms, agent.VOICE_BLOCK_MS)
        # So the pause is counted from the word, not from the recording.
        for _ in range(agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS - 1):
            watcher.add(0.0, block)
        self.assertTrue(watcher.paused)

    def test_room_noise_does_not_hide_the_pause(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        # A noisy room, a word over the noise, and then the same noise again:
        # however loud the room is, the pause behind it is still a pause.
        for _ in range(agent.VOICE_NOISE_BLOCKS):
            watcher.add(0.05, block)
        self.assertEqual(watcher.noise, 0.05)
        self.assertGreater(watcher.threshold, 0.05)  # the noise stays silence
        watcher.add(0.2, block)  # the word
        self.assertTrue(watcher.spoken)
        for _ in range(agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS - 1):
            watcher.add(0.05, block)  # still a block short of a full pause
            self.assertFalse(watcher.paused)
        watcher.add(0.05, block)
        self.assertTrue(watcher.paused)

    def test_a_dropout_does_not_pass_for_a_silent_room(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        for _ in range(agent.VOICE_NOISE_BLOCKS):
            watcher.add(0.05, block)
        watcher.add(0.2, block)  # the word
        # A block that drops to nothing in the middle of the pause must not
        # be taken for a silent room, which would turn the noise into speech.
        for i in range(agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS):
            watcher.add(0.0 if i in (5, 12) else 0.05, block)
        self.assertTrue(watcher.paused)
        self.assertEqual(watcher.noise, 0.05)
        self.assertGreater(watcher.threshold, 0.05)

    def test_quiet_microphone_still_hears_the_pause(self) -> None:
        block = agent.VOICE_SAMPLE_RATE * agent.VOICE_BLOCK_MS // 1000
        watcher = agent.PauseWatcher()
        # The other end of the scale: a mic so quiet that speech sits at a
        # thousandth of full scale still has to be told apart from the room.
        for _ in range(agent.VOICE_NOISE_BLOCKS):
            watcher.add(0.0004, block)
        watcher.add(0.006, block)
        self.assertTrue(watcher.spoken)
        self.assertLess(watcher.threshold, 0.006)
        for _ in range(agent.VOICE_SILENCE_MS // agent.VOICE_BLOCK_MS + 1):
            watcher.add(0.0004, block)
        self.assertTrue(watcher.paused)

    def test_voice_prompt_audio_is_a_mono_16khz_wav(self) -> None:
        import io as fake_io
        import wave as fake_wave

        import numpy as np

        samples = np.zeros(agent.VOICE_SAMPLE_RATE, dtype="float32")
        with fake_wave.open(fake_io.BytesIO(agent.encode_wav(samples))) as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), agent.VOICE_SAMPLE_RATE)
            self.assertEqual(wav.getnframes(), len(samples))

    def test_a_quiet_recording_is_turned_up_before_it_is_sent(self) -> None:
        import io as fake_io
        import wave as fake_wave

        import numpy as np

        def peak_of(samples: object) -> float:
            with fake_wave.open(fake_io.BytesIO(agent.encode_wav(samples))) as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            return float(np.max(np.abs(pcm))) / 32767

        # Speech spoken quietly is a whisper the model guesses at, so it is
        # brought up to a level it can transcribe.
        quiet = np.full(1600, 0.01, dtype="float32")
        self.assertAlmostEqual(peak_of(quiet), agent.VOICE_PEAK, delta=0.01)
        # A recording of nothing but the room is not amplified without bound.
        hiss = np.full(1600, 0.0001, dtype="float32")
        self.assertLessEqual(peak_of(hiss), agent.VOICE_MAX_GAIN * 0.0001 + 0.01)
        # A recording that is already loud enough is left alone.
        loud = np.full(1600, 0.99, dtype="float32")
        self.assertAlmostEqual(peak_of(loud), 0.99, delta=0.01)
        # Silence stays silence, rather than being turned into full-scale hiss.
        self.assertEqual(peak_of(np.zeros(1600, dtype="float32")), 0.0)

    def test_transcription_asks_the_model_and_skips_thinking(self) -> None:
        with fake_voice_modules():
            self._run_agent(
                [agent.VOICE_COMMAND, "", ""],
                responses=["typed out loud", "REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        first = FakeOllamaHandler.payloads[0]
        self.assertEqual(first["model"], "test-model")
        self.assertEqual(first["reasoning_effort"], "none")
        parts = first["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[0]["text"], agent.TRANSCRIBE_PROMPT)
        self.assertEqual(parts[1]["type"], "input_audio")

    @unittest.skipIf(agent.readline is None, "readline is not available")
    def test_voice_prompt_kept_in_input_history(self) -> None:
        with fake_voice_modules():
            self._run_agent(
                [agent.VOICE_COMMAND, "", ""],
                responses=["remember me", "REPLY"],
                markers=[b"You: ", b"Recording...", b"You: "],
            )
        history = Path(agent.HISTORY_FILE).read_text(encoding="utf-8")
        self.assertIn("remember me", history)

    def test_voice_prompt_without_packages_reports_error(self) -> None:
        with fake_voice_modules(broken=("sounddevice",)):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("pip install sounddevice", output)
        self.assertIn("Assistant: REPLY", output)
        # The failed voice prompt is not sent, the dialog keeps working.
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_voice_prompt_with_broken_microphone_reports_error(self) -> None:
        with fake_voice_modules(mic_error=True):
            output = self._run_agent(
                [agent.VOICE_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("no default input device", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_voice_prompt_with_failing_transcription_reports_error(self) -> None:
        FakeOllamaHandler.audio_error = "audio is not supported"
        self.addCleanup(setattr, FakeOllamaHandler, "audio_error", None)
        with fake_voice_modules():
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", "hello", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"Recording...", b"You: ", b"You: "],
            )
        self.assertIn("voice input failed (HTTP Error 500", output)
        self.assertIn("Assistant: REPLY", output)
        # The prompt with no transcript is never sent, the dialog keeps going.
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_silent_voice_prompt_is_ignored(self) -> None:
        with fake_voice_modules():
            output = self._run_agent(
                [agent.VOICE_COMMAND, "", "hello", ""],
                responses=["", "REPLY"],
                markers=[b"You: ", b"Recording...", b"You: ", b"You: "],
            )
        self.assertIn("Nothing recognized", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(self._last_user_prompts(), ["hello"])

    def test_reply_spoken_by_default(self) -> None:
        agent.SPEAK_DEFAULT = True
        with fake_voice_modules() as calls:
            output = self._run_agent(["hello", ""], responses=["REPLY"])
        self.assertIn("Voice output on", output)
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(calls["spoken"], [(agent.PIPER_VOICE_EN, "REPLY")])
        self.assertEqual(len(calls["played"]), 1)

    def test_reply_not_spoken_when_voice_output_is_off(self) -> None:
        with fake_voice_modules() as calls:
            output = self._run_agent(["hello", ""], responses=["REPLY"])
        self.assertIn("Assistant: REPLY", output)
        self.assertEqual(calls["spoken"], [])
        self.assertEqual(calls["played"], [])

    def test_reply_spoken_after_speak_toggle(self) -> None:
        with fake_voice_modules() as calls:
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "hello", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"You: ", b"You: "],
            )
        self.assertIn(agent.PIPER_VOICE_EN, output)
        self.assertIn(agent.PIPER_VOICE_RU, output)
        self.assertIn("Assistant: REPLY", output)
        self.assertIn("Bye!", output)
        self.assertEqual(calls["spoken"], [(agent.PIPER_VOICE_EN, "REPLY")])
        self.assertEqual(len(calls["played"]), 1)
        self.assertEqual(calls["played"][0][1], 22050)

    def test_speak_toggle_mutes_again(self) -> None:
        with fake_voice_modules() as calls:
            output = self._run_agent(
                [agent.SPEAK_COMMAND, agent.SPEAK_COMMAND, "hello", ""],
                responses=["REPLY"],
                markers=[b"You: ", b"You: ", b"You: ", b"You: "],
            )
        self.assertIn("Voice output off.", output)
        self.assertEqual(calls["spoken"], [])

    def test_voice_played_for_every_reply_while_on(self) -> None:
        with fake_voice_modules() as calls:
            self._run_agent(
                [agent.SPEAK_COMMAND, "one", "two", ""],
                responses=["FIRST", "SECOND"],
                markers=[b"You: "] * 5,
            )
        self.assertEqual(
            calls["spoken"],
            [(agent.PIPER_VOICE_EN, "FIRST"), (agent.PIPER_VOICE_EN, "SECOND")],
        )

    def test_ctrl_c_while_speaking_stops_the_voice_and_keeps_the_dialog(
        self,
    ) -> None:
        with fake_voice_modules(interrupt_wait=True) as calls:
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "hello", "again", ""],
                responses=["FIRST", "SECOND"],
                markers=[b"You: "] * 4,
            )
        self.assertIn("Assistant: FIRST", output)
        self.assertIn("Assistant: SECOND", output)
        # Every reply has its speech cut short, and the dialog goes on.
        self.assertEqual(output.count("Voice stopped."), 2)
        self.assertEqual(len(calls["played"]), 2)
        self.assertIn("Bye!", output)

    def test_russian_reply_spoken_with_russian_voice(self) -> None:
        with fake_voice_modules() as calls:
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "привет", ""],
                responses=["Привет! Как дела?"],
                markers=[b"You: "] * 3,
            )
        self.assertIn("Assistant: Привет!", output)
        self.assertEqual(calls["spoken"], [(agent.PIPER_VOICE_RU, "Привет! Как дела?")])

    def test_only_the_language_actually_spoken_is_downloaded(self) -> None:
        with fake_voice_modules() as calls:
            self._run_agent(
                [agent.SPEAK_COMMAND, "one", "два", ""],
                responses=["English answer", "Русский ответ"],
                markers=[b"You: "] * 5,
            )
        self.assertEqual(
            calls["downloaded"],
            [
                (agent.PIPER_VOICE_EN, agent.TTS_DIR),
                (agent.PIPER_VOICE_RU, agent.TTS_DIR),
            ],
        )

    def test_voice_downloaded_once_and_cached(self) -> None:
        with fake_voice_modules() as calls:
            self._run_agent(
                [agent.SPEAK_COMMAND, "one", "two", ""],
                responses=["FIRST", "SECOND"],
                markers=[b"You: "] * 5,
            )
            self.assertEqual(
                calls["downloaded"], [(agent.PIPER_VOICE_EN, agent.TTS_DIR)]
            )
            self.assertEqual(len(calls["loaded"]), 1)
            model_path, config_path = calls["loaded"][0]
            self.assertTrue(model_path.endswith(f"{agent.PIPER_VOICE_EN}.onnx"))
            self.assertTrue(config_path.endswith(f"{agent.PIPER_VOICE_EN}.onnx.json"))
            # A second session reuses the files already on disk.
            calls["downloaded"].clear()
            self._run_agent(
                [agent.SPEAK_COMMAND, "hello", ""],
                responses=["REPLY"],
                markers=[b"You: "] * 3,
            )
        self.assertEqual(calls["downloaded"], [])
        self.assertEqual(
            calls["spoken"],
            [
                (agent.PIPER_VOICE_EN, "FIRST"),
                (agent.PIPER_VOICE_EN, "SECOND"),
                (agent.PIPER_VOICE_EN, "REPLY"),
            ],
        )

    def test_detect_language(self) -> None:
        self.assertEqual(agent.detect_language("Hello there"), "en")
        self.assertEqual(agent.detect_language("Привет, как дела?"), "ru")
        # The script the reply mostly uses wins, whatever the other one is.
        self.assertEqual(agent.detect_language("Ответ про Python 3.11 и списки"), "ru")
        self.assertEqual(agent.detect_language("Answer mentions «Привет» once"), "en")
        self.assertEqual(agent.detect_language("1234 - 5678 = ?"), "en")

    def test_speakable_text_strips_markup(self) -> None:
        self.assertEqual(
            agent.speakable_text(
                "Here is `print(1)` and [docs](https://example.com/x) plus "
                "https://example.org/y\n\n```python\nprint('nope')\n```\n**Done!**"
            ),
            "Here is print(1) and docs plus link Done!",
        )

    def test_speakable_text_link_word_follows_language(self) -> None:
        self.assertEqual(
            agent.speakable_text(
                "Смотри https://example.com", agent.TTS_LINK_WORDS["ru"]
            ),
            "Смотри ссылка",
        )

    def test_speakable_text_caps_long_reply(self) -> None:
        spoken = agent.speakable_text("word " * agent.TTS_MAX_CHARS)
        self.assertLessEqual(len(spoken), agent.TTS_MAX_CHARS + len(", and more."))
        self.assertTrue(spoken.endswith(", and more."))

    def test_voice_output_without_packages_reports_error(self) -> None:
        with fake_voice_modules(broken=("piper", "piper.download_voices")):
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("pip install piper-tts sounddevice", output)
        self.assertIn("Assistant: REPLY", output)
        # Voice output stays off, the dialog keeps working.
        self.assertNotIn("Voice output on", output)

    def test_voice_output_with_broken_model_reports_error(self) -> None:
        with fake_voice_modules(tts_error=True):
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "hello", ""], responses=["REPLY"]
            )
        self.assertIn("cannot enable voice output (broken onnx model)", output)
        self.assertIn("Assistant: REPLY", output)

    def test_reply_with_nothing_speakable_is_not_played(self) -> None:
        with fake_voice_modules() as calls:
            output = self._run_agent(
                [agent.SPEAK_COMMAND, "hello", ""],
                responses=["```\nprint('hi')\n```"],
                markers=[b"You: "] * 3,
            )
        self.assertIn("Assistant:", output)
        self.assertEqual(calls["spoken"], [])
        self.assertEqual(calls["played"], [])

    def test_translate_command_sends_its_prompt(self) -> None:
        with fake_voice_modules():
            output = self._run_agent(
                [agent.TRANSLATE_COMMAND, "what is the weather", ""],
                responses=["FIRST", "SECOND"],
            )
        self.assertIn("Assistant: FIRST", output)
        self.assertIn("Assistant: SECOND", output)
        self.assertEqual(
            self._last_user_prompts(),
            [agent.TRANSLATE_PROMPT, "what is the weather"],
        )

    def test_translate_command_is_kept_in_input_history(self) -> None:
        with fake_voice_modules():
            self._run_agent([agent.TRANSLATE_COMMAND, ""], responses=["REPLY"])
        history = Path(agent.HISTORY_FILE).read_text(encoding="utf-8")
        self.assertIn(agent.TRANSLATE_COMMAND, history)

    def test_translate_command_ru_sends_its_prompt(self) -> None:
        with fake_voice_modules():
            output = self._run_agent(
                [agent.TRANSLATE_COMMAND_RU, "какая погода", ""],
                responses=["FIRST", "SECOND"],
            )
        self.assertIn("Assistant: FIRST", output)
        self.assertIn("Assistant: SECOND", output)
        self.assertEqual(
            self._last_user_prompts(),
            [agent.TRANSLATE_PROMPT_RU, "какая погода"],
        )


if __name__ == "__main__":
    unittest.main()
