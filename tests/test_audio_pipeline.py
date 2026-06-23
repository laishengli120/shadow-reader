import base64
import io
import struct
import threading
import time
import unittest
import wave
from unittest import mock

from pydub import AudioSegment

import app as shadow


def _wav_bytes(amplitude: int, duration_ms: int = 40, frame_rate: int = 8000) -> bytes:
    sample_count = int(frame_rate * duration_ms / 1000)
    frames = struct.pack("<h", amplitude) * sample_count
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


class FakeProvider(shadow.TTSProvider):
    voices = [
        shadow.VoiceOption("zh", "Chinese", "zh"),
        shadow.VoiceOption("en", "English", "en"),
    ]
    audio_format = "wav"
    supports_native_rate = True
    max_workers = 8

    def __init__(
        self,
        *,
        delay: float = 0.0,
        max_workers: int = 8,
        supports_native_rate: bool = True,
        amplitudes: dict[str, int] | None = None,
    ) -> None:
        self.delay = delay
        self.max_workers = max_workers
        self.supports_native_rate = supports_native_rate
        self.amplitudes = amplitudes or {}
        self.calls: list[tuple[str, str, float]] = []
        self.active = 0
        self.peak_active = 0
        self._lock = threading.Lock()

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        with self._lock:
            self.calls.append((text, voice, rate))
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            amplitude = self.amplitudes.get(
                text,
                1000 + sum(ord(char) for char in text) % 20000,
            )
            return _wav_bytes(amplitude)
        finally:
            with self._lock:
                self.active -= 1


class AudioPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        shadow._tts_cache_clear()

    def test_mixed_language_parts_run_concurrently_and_keep_order(self) -> None:
        amplitudes = {"中": 1000, "A": 2000, "文": 3000, "B": 4000}
        provider = FakeProvider(delay=0.05, amplitudes=amplitudes)

        segments, metrics = shadow._synthesize_lines(
            provider,
            ["中A文B"],
            "zh",
            "en",
            1.0,
        )

        self.assertGreaterEqual(provider.peak_active, 2)
        self.assertEqual(metrics.worker_count, 4)
        self.assertEqual(metrics.part_count, 4)
        self.assertEqual(len(segments[0]), 160)

        samples = segments[0].get_array_of_samples()
        samples_per_part = int(segments[0].frame_rate * 0.04)
        centers = [samples_per_part * index + samples_per_part // 2 for index in range(4)]
        self.assertEqual([samples[index] for index in centers], [1000, 2000, 3000, 4000])
        self.assertEqual(
            {(text, voice) for text, voice, _ in provider.calls},
            {("中", "zh"), ("A", "en"), ("文", "zh"), ("B", "en")},
        )

    def test_cloud_worker_count_is_capped_at_eight(self) -> None:
        provider = FakeProvider(delay=0.04, max_workers=8)

        _, metrics = shadow._synthesize_lines(
            provider,
            ["中A文B国C语D言E"],
            "zh",
            "en",
            1.0,
        )

        self.assertEqual(metrics.part_count, 10)
        self.assertEqual(metrics.worker_count, 8)
        self.assertEqual(provider.peak_active, 8)

    def test_local_providers_are_single_worker(self) -> None:
        provider = FakeProvider(delay=0.01, max_workers=1)

        _, metrics = shadow._synthesize_lines(
            provider,
            ["中A文B"],
            "zh",
            "en",
            1.0,
        )

        self.assertEqual(metrics.worker_count, 1)
        self.assertEqual(provider.peak_active, 1)
        self.assertEqual(shadow.Pyttsx3Provider.max_workers, 1)

    def test_gtts_internal_requests_run_concurrently_and_keep_order(self) -> None:
        class FakeGTTS:
            timeout = 10.0

            def __init__(self, **kwargs) -> None:
                self.timeout = kwargs["timeout"]

            def _prepare_requests(self):
                return ["part-0", "part-1", "part-2", "part-3"]

        active = 0
        peak_active = 0
        lock = threading.Lock()

        def fake_send(tts_obj, prepared_request):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                time.sleep(0.03)
                return prepared_request.encode()
            finally:
                with lock:
                    active -= 1

        provider = shadow.GTTSProvider()
        provider._chunk_workers = 4

        with (
            mock.patch("gtts.gTTS", FakeGTTS),
            mock.patch.object(
                shadow.GTTSProvider,
                "_send_prepared_request",
                staticmethod(fake_send),
            ),
        ):
            audio = provider.tts("hello world", "en")

        self.assertEqual(audio, b"part-0part-1part-2part-3")
        self.assertGreaterEqual(peak_active, 2)

    def test_repeated_lines_are_cached_not_re_synthesized(self) -> None:
        provider = FakeProvider(delay=0.03)

        segments, metrics = shadow._synthesize_lines(
            provider,
            ["hello", "hello", "hello", "hello"],
            "zh",
            "en",
            1.0,
        )

        self.assertEqual(len(segments), 4)
        self.assertEqual(metrics.part_count, 4)
        self.assertEqual(metrics.unique_part_count, 1)
        self.assertEqual(metrics.deduped_part_count, 3)
        self.assertEqual(metrics.worker_count, 1)
        # 同一批任务先去重，避免重复项并发 miss 后重复请求 TTS。
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][0], "hello")

    def test_numbered_repeated_lines_share_spoken_text(self) -> None:
        provider = FakeProvider(delay=0.03)

        segments, metrics = shadow._synthesize_lines(
            provider,
            ["1. hello", "hello"],
            "zh",
            "en",
            1.0,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(metrics.part_count, 2)
        self.assertEqual(metrics.unique_part_count, 1)
        self.assertEqual(metrics.deduped_part_count, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][0], "hello")

    def test_rate_post_processing_only_runs_for_non_native_provider(self) -> None:
        native_provider = FakeProvider(supports_native_rate=True)
        non_native_provider = FakeProvider(supports_native_rate=False)

        with mock.patch.object(
            shadow,
            "_adjust_audio_rate",
            side_effect=lambda segment, rate: segment,
        ) as adjust_rate:
            shadow._synthesize_lines(native_provider, ["one", "two"], "zh", "en", 1.2)
            self.assertEqual(adjust_rate.call_count, 0)

            shadow._synthesize_lines(non_native_provider, ["one", "two"], "zh", "en", 1.2)
            self.assertEqual(adjust_rate.call_count, 2)
            for call in adjust_rate.call_args_list:
                self.assertEqual(call.args[1], 1.2)

    def test_post_processed_rate_changes_audio_duration(self) -> None:
        original = AudioSegment.silent(duration=600)

        slower = shadow._adjust_audio_rate(original, 0.8)
        faster = shadow._adjust_audio_rate(original, 1.5)

        self.assertGreater(len(slower), len(original))
        self.assertLess(len(faster), len(original))


class GenerateEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_limiter_state = shadow.limiter.enabled
        shadow.limiter.enabled = False
        self.client = shadow.app.test_client()

    def tearDown(self) -> None:
        shadow.limiter.enabled = self.previous_limiter_state

    @staticmethod
    def _request_body(text: str) -> dict:
        return {
            "api_key": "valid-test-key",
            "text": text,
            "provider": "openai",
            "voice": "en",
            "voice_zh": "zh",
            "voice_en": "en",
            "interval": 0.1,
            "speech_rate": 1.0,
        }

    def test_response_is_compatible_playable_and_encoded_once(self) -> None:
        provider = FakeProvider()
        original_export = AudioSegment.export
        mp3_exports = 0

        def counting_export(segment, *args, **kwargs):
            nonlocal mp3_exports
            if kwargs.get("format") == "mp3":
                mp3_exports += 1
            return original_export(segment, *args, **kwargs)

        with (
            mock.patch.object(shadow.ProviderRegistry, "get", return_value=provider),
            mock.patch.object(AudioSegment, "export", new=counting_export),
        ):
            response = self.client.post(
                "/generate",
                json=self._request_body("中A\nhello"),
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(set(payload), {"audio_base64", "timings", "metrics"})
        self.assertEqual(mp3_exports, 1)

        audio = base64.b64decode(payload["audio_base64"])
        decoded = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
        self.assertGreater(len(decoded), 0)

        timings = payload["timings"]
        self.assertEqual([timing["text"] for timing in timings], ["中A", "hello"])
        self.assertLess(timings[0]["start_time"], timings[0]["end_time"])
        self.assertLessEqual(timings[0]["end_time"], timings[1]["start_time"])
        self.assertLess(timings[1]["start_time"], timings[1]["end_time"])

        metrics = payload["metrics"]
        self.assertEqual(metrics["part_count"], 3)
        self.assertEqual(metrics["worker_count"], 3)
        self.assertGreaterEqual(metrics["total_seconds"], 0)
        self.assertGreaterEqual(metrics["synthesis_seconds"], 0)

    def test_empty_and_over_limit_text_are_rejected(self) -> None:
        with mock.patch.object(
            shadow.ProviderRegistry,
            "get",
            return_value=FakeProvider(),
        ):
            empty_response = self.client.post(
                "/generate",
                json=self._request_body("  \n "),
            )
            over_limit_response = self.client.post(
                "/generate",
                json=self._request_body("\n".join(["line"] * 51)),
            )

        self.assertEqual(empty_response.status_code, 400)
        self.assertIn("不能为空", empty_response.get_json()["error"])
        self.assertEqual(over_limit_response.status_code, 400)
        self.assertIn("50", over_limit_response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
