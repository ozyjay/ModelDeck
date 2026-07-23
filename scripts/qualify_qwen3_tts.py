from __future__ import annotations

import argparse
import io
import json
import re
import threading
import time
import unicodedata
import wave
from array import array
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import torch
from modeldeck.speechshift import QWEN_TTS_GENERATION_TIMEOUT_SECONDS

FIXED_INPUTS = {
    "en": "Welcome to JCU Open Day. Speech Shift is running locally.",
    "fr": "Bienvenue à la journée portes ouvertes de JCU. Speech Shift fonctionne localement.",
    "de": "Willkommen zum Tag der offenen Tür der JCU. Speech Shift läuft lokal.",
}
SAMPLE_RATE_HZ = 24_000
TRANSCRIPTION_SAMPLE_RATE_HZ = 16_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physically qualify the deployed Qwen3-TTS Worker without retaining speech."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8668")
    parser.add_argument("--model", required=True)
    parser.add_argument("--whisper-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_loopback(args.endpoint)
    if not args.whisper_snapshot.is_dir():
        raise SystemExit("The pinned multilingual Whisper snapshot is unavailable.")

    monitor = DeviceMemoryMonitor()
    monitor.start()
    audio_by_language: dict[str, np.ndarray] = {}
    measurements: list[dict[str, object]] = []
    try:
        with httpx.Client(base_url=args.endpoint, timeout=90) as client:
            runtime = client.get("/metrics").raise_for_status().json()
            model = client.get("/model").raise_for_status().json()
            for language, text in FIXED_INPUTS.items():
                started = time.perf_counter()
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "request_id": f"qualification-{language}-20260723",
                        "model": args.model,
                        "input": text,
                        "voice": "ryan",
                        "language": language,
                        "response_format": "wav",
                    },
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                samples, sample_rate = _decode_wav(response.content)
                if sample_rate != SAMPLE_RATE_HZ:
                    raise RuntimeError("The Worker returned an unexpected sample rate.")
                metrics = client.get("/metrics").raise_for_status().json()
                last_request = metrics["last_request"]
                duration = len(samples) / sample_rate
                clipped = int(np.count_nonzero((samples == -32_768) | (samples == 32_767)))
                audio_by_language[language] = samples.astype(np.float32) / 32_768
                measurements.append(
                    {
                        "language": language,
                        "generation_seconds": last_request["inference_seconds"],
                        "total_worker_seconds": last_request["total_worker_seconds"],
                        "request_seconds": round(elapsed, 6),
                        "output_duration_seconds": round(duration, 6),
                        "real_time_factor": round(elapsed / duration, 6),
                        "first_audio_latency_seconds": round(elapsed, 6),
                        "clipped_sample_percent": round(clipped / max(1, len(samples)) * 100, 9),
                        "peak_temperatures_c": metrics["last_temperatures"],
                    }
                )
                text = ""
                response = None

        transcriptions = _transcription_measurements(args.whisper_snapshot, audio_by_language)
        for measurement in measurements:
            measurement["transcription"] = transcriptions[measurement["language"]]
    finally:
        monitor.stop()
        for samples in audio_by_language.values():
            samples.fill(0)
        audio_by_language.clear()

    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "dtype": model["dtype"],
            "attention_implementation": runtime["attention_implementation"],
            "do_sample": True,
            "subtalker_dosample": True,
            "maximum_codec_tokens": 256,
            "generation_timeout_seconds": QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
            "resident_and_warmed": True,
            "streaming": False,
        },
        "runtime": {
            key: runtime[key]
            for key in (
                "device_name",
                "torch_version",
                "hip_version",
                "qwen_tts_version",
                "transformers_version",
            )
        },
        "measurements": measurements,
        "global_device_memory": {
            "baseline_used_mb": monitor.baseline_used_mb,
            "peak_used_mb": monitor.peak_used_mb,
            "peak_delta_mb": round(monitor.peak_used_mb - monitor.baseline_used_mb, 3),
        },
        "privacy": {
            "fixed_synthetic_inputs": True,
            "audio_retained": False,
            "transcripts_retained": False,
            "content_in_report": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


class DeviceMemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_used_mb = 0.0
        self.peak_used_mb = 0.0

    def start(self) -> None:
        self.baseline_used_mb = self._used_mb()
        self.peak_used_mb = self.baseline_used_mb
        self._thread = threading.Thread(target=self._run, name="qwen-qualification-memory", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self.peak_used_mb = max(self.peak_used_mb, self._used_mb())

    @staticmethod
    def _used_mb() -> float:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return round((total_bytes - free_bytes) / 1024**2, 3)


def _decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise RuntimeError("The Worker returned an unexpected WAV format.")
        sample_rate = reader.getframerate()
        samples = array("h")
        samples.frombytes(reader.readframes(reader.getnframes()))
    return np.asarray(samples, dtype=np.int16), sample_rate


def _transcription_measurements(
    snapshot: Path,
    audio_by_language: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    results: dict[str, dict[str, object]] = {}
    try:
        for language, audio in audio_by_language.items():
            resampled = _resample(audio, SAMPLE_RATE_HZ, TRANSCRIPTION_SAMPLE_RATE_HZ)
            inputs = processor(
                resampled,
                sampling_rate=TRANSCRIPTION_SAMPLE_RATE_HZ,
                return_tensors="pt",
            )
            with torch.inference_mode():
                generated = model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe",
                    do_sample=False,
                )
            transcript = processor.batch_decode(generated, skip_special_tokens=True)[0]
            reference_words = _normalised_words(FIXED_INPUTS[language])
            transcript_words = _normalised_words(transcript)
            distance = _edit_distance(reference_words, transcript_words)
            results[language] = {
                "word_error_rate": round(distance / max(1, len(reference_words)), 6),
                "normalised_exact_match": reference_words == transcript_words,
                "reference_word_count": len(reference_words),
                "transcript_word_count": len(transcript_words),
            }
            transcript = ""
            resampled.fill(0)
    finally:
        del model
        del processor
    return results


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_length = round(len(samples) * target_rate / source_rate)
    source_positions = np.arange(len(samples), dtype=np.float64)
    target_positions = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _normalised_words(value: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"\w+", normalised, flags=re.UNICODE)


def _edit_distance(reference: list[str], candidate: list[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_word in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1] + (reference_word != candidate_word),
                )
            )
        previous = current
    return previous[-1]


def _require_loopback(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in {"", "/"}:
        raise SystemExit("The qualification endpoint must be an HTTP loopback origin.")


if __name__ == "__main__":
    raise SystemExit(main())
