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
from statistics import median
from urllib.parse import urlparse

import httpx
import numpy as np
import torch
from modeldeck.speechshift import (
    QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
    QWEN_TTS_MAXIMUM_CODEC_TOKENS,
    QWEN_TTS_SAMPLE_RATE_HZ,
    QWEN_TTS_VOICES,
)

FIXED_INPUTS = {
    "en": "Welcome to JCU Open Day. Speech Shift is running locally.",
    "fr": "Bienvenue à la journée portes ouvertes de JCU. Speech Shift fonctionne localement.",
    "de": "Willkommen zum Tag der offenen Tür der JCU. Speech Shift läuft lokal.",
}
SAMPLE_RATE_HZ = QWEN_TTS_SAMPLE_RATE_HZ
TRANSCRIPTION_SAMPLE_RATE_HZ = 16_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physically qualify the deployed Qwen3-TTS Worker without retaining speech."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8668")
    parser.add_argument("--model", required=True)
    parser.add_argument("--whisper-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--voices",
        nargs="+",
        choices=QWEN_TTS_VOICES,
        default=["vivian", "serena"],
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=FIXED_INPUTS,
        default=list(FIXED_INPUTS),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=0)
    parser.add_argument("--sample-output-dir", type=Path)
    args = parser.parse_args()
    _require_loopback(args.endpoint)
    if not args.whisper_snapshot.is_dir():
        raise SystemExit("The pinned multilingual Whisper snapshot is unavailable.")
    if not 1 <= args.repetitions <= 10:
        raise SystemExit("Repetitions must be between 1 and 10.")
    if not 0 <= args.cooldown_seconds <= 120:
        raise SystemExit("Cooldown seconds must be between 0 and 120.")
    voices = tuple(dict.fromkeys(args.voices))
    languages = tuple(dict.fromkeys(args.languages))

    monitor = DeviceMemoryMonitor()
    monitor.start()
    audio_by_sample: dict[tuple[str, str, int], np.ndarray] = {}
    measurements: list[dict[str, object]] = []
    retained_samples: list[str] = []
    completed_requests = 0
    try:
        with httpx.Client(base_url=args.endpoint, timeout=90) as client:
            runtime = client.get("/metrics").raise_for_status().json()
            model = client.get("/model").raise_for_status().json()
            for voice in voices:
                for language in languages:
                    fixed_text = FIXED_INPUTS[language]
                    for repetition in range(1, args.repetitions + 1):
                        if completed_requests and args.cooldown_seconds:
                            print(
                                f"Cooling for {args.cooldown_seconds:g} seconds before "
                                f"{voice}/{language} repetition {repetition}",
                                flush=True,
                            )
                            time.sleep(args.cooldown_seconds)
                        request_text = fixed_text
                        thermal_retries = 0
                        while True:
                            request_id = f"qual-{voice}-{language}-{repetition}-{thermal_retries}-20260723"
                            monitor.begin_window()
                            started = time.perf_counter()
                            response = client.post(
                                "/v1/audio/speech",
                                json={
                                    "request_id": request_id,
                                    "model": args.model,
                                    "input": request_text,
                                    "voice": voice,
                                    "language": language,
                                    "response_format": "wav",
                                },
                            )
                            elapsed = time.perf_counter() - started
                            memory = monitor.end_window()
                            error_code = _error_code(response)
                            if (
                                response.status_code != 200
                                and error_code == "thermal_cooldown_required"
                                and thermal_retries < 12
                            ):
                                thermal_retries += 1
                                retry_delay = max(5, args.cooldown_seconds)
                                print(
                                    f"{voice}/{language} repetition {repetition}: "
                                    f"thermal cooldown, retrying in {retry_delay:g} seconds",
                                    flush=True,
                                )
                                response = None
                                time.sleep(retry_delay)
                                continue
                            break
                        request_text = ""
                        if response.status_code != 200:
                            measurements.append(
                                {
                                    "voice": voice,
                                    "language": language,
                                    "repetition": repetition,
                                    "outcome": "failure",
                                    "request_seconds": round(elapsed, 6),
                                    "error_code": error_code,
                                    "device_memory": memory,
                                    "thermal_admission_retries": thermal_retries,
                                }
                            )
                            print(
                                f"{voice}/{language} repetition {repetition}: "
                                f"failed ({error_code}) after {elapsed:.3f} seconds",
                                flush=True,
                            )
                            completed_requests += 1
                            response = None
                            continue
                        wav_payload = response.content
                        samples, sample_rate = _decode_wav(wav_payload)
                        if sample_rate != SAMPLE_RATE_HZ:
                            raise RuntimeError("The Worker returned an unexpected sample rate.")
                        metrics = client.get("/metrics").raise_for_status().json()
                        last_request = metrics["last_request"]
                        duration = len(samples) / sample_rate
                        clipped = int(np.count_nonzero((samples == -32_768) | (samples == 32_767)))
                        normalised_samples = samples.astype(np.float32) / 32_768
                        audio_by_sample[(voice, language, repetition)] = normalised_samples
                        if args.sample_output_dir is not None and repetition == 1:
                            args.sample_output_dir.mkdir(parents=True, exist_ok=True)
                            sample_path = args.sample_output_dir / f"{voice}-{language}.wav"
                            sample_path.write_bytes(wav_payload)
                            retained_samples.append(sample_path.name)
                        measurements.append(
                            {
                                "voice": voice,
                                "language": language,
                                "repetition": repetition,
                                "outcome": "success",
                                "generation_seconds": last_request["inference_seconds"],
                                "total_worker_seconds": last_request["total_worker_seconds"],
                                "request_seconds": round(elapsed, 6),
                                "output_duration_seconds": round(duration, 6),
                                "real_time_factor": round(elapsed / duration, 6),
                                "first_audio_latency_seconds": round(elapsed, 6),
                                "clipped_sample_count": clipped,
                                "clipped_sample_percent": round(clipped / max(1, len(samples)) * 100, 9),
                                "peak_absolute_amplitude": round(
                                    float(np.max(np.abs(normalised_samples))), 9
                                ),
                                "rms_amplitude": round(
                                    float(np.sqrt(np.mean(np.square(normalised_samples)))),
                                    9,
                                ),
                                "wav_validation": {
                                    "sample_rate_hz": sample_rate,
                                    "channels": 1,
                                    "sample_width_bytes": 2,
                                    "valid": True,
                                },
                                "peak_temperatures_c": metrics["last_temperatures"],
                                "device_memory": memory,
                                "thermal_admission_retries": thermal_retries,
                            }
                        )
                        print(
                            f"{voice}/{language} repetition {repetition}: passed in {elapsed:.3f} seconds",
                            flush=True,
                        )
                        completed_requests += 1
                        wav_payload = b""
                        response = None

        transcriptions = _transcription_measurements(args.whisper_snapshot, audio_by_sample)
        for measurement in measurements:
            if measurement["outcome"] == "success":
                key = (
                    str(measurement["voice"]),
                    str(measurement["language"]),
                    int(measurement["repetition"]),
                )
                measurement["transcription"] = transcriptions[key]
    finally:
        monitor.stop()
        for samples in audio_by_sample.values():
            samples.fill(0)
        audio_by_sample.clear()

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
            "maximum_codec_tokens": QWEN_TTS_MAXIMUM_CODEC_TOKENS,
            "generation_timeout_seconds": QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
            "resident_and_warmed": True,
            "streaming": False,
            "curated_voices": list(QWEN_TTS_VOICES),
            "qualified_voices": list(voices),
            "languages": list(languages),
            "repetitions_per_voice_language": args.repetitions,
            "cooldown_seconds": args.cooldown_seconds,
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
        "qualification_groups": _aggregate_measurements(measurements),
        "request_summary": {
            "total": len(measurements),
            "successful": sum(item["outcome"] == "success" for item in measurements),
            "failed": sum(item["outcome"] == "failure" for item in measurements),
        },
        "global_device_memory": {
            "baseline_used_mb": monitor.baseline_used_mb,
            "peak_used_mb": monitor.peak_used_mb,
            "peak_delta_mb": round(monitor.peak_used_mb - monitor.baseline_used_mb, 3),
        },
        "privacy": {
            "fixed_synthetic_inputs": True,
            "audio_retained": bool(retained_samples),
            "retained_fixed_sample_files": sorted(retained_samples),
            "transcripts_retained": False,
            "content_in_report": False,
            "visitor_content_processed": False,
        },
        "manual_review": {
            "status": "pending",
            "criteria": ["pronunciation", "intelligibility"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return int(report["request_summary"]["failed"] != 0)


class DeviceMemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._window_active = False
        self._window_baseline_used_mb = 0.0
        self._window_peak_used_mb = 0.0
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

    def begin_window(self) -> None:
        used_mb = self._used_mb()
        with self._lock:
            self._window_baseline_used_mb = used_mb
            self._window_peak_used_mb = used_mb
            self._window_active = True

    def end_window(self) -> dict[str, float]:
        used_mb = self._used_mb()
        with self._lock:
            self._window_peak_used_mb = max(self._window_peak_used_mb, used_mb)
            baseline = self._window_baseline_used_mb
            peak = self._window_peak_used_mb
            self._window_active = False
        return {
            "baseline_used_mb": baseline,
            "peak_used_mb": peak,
            "peak_delta_mb": round(peak - baseline, 3),
        }

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            used_mb = self._used_mb()
            with self._lock:
                self.peak_used_mb = max(self.peak_used_mb, used_mb)
                if self._window_active:
                    self._window_peak_used_mb = max(self._window_peak_used_mb, used_mb)

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
    audio_by_sample: dict[tuple[str, str, int], np.ndarray],
) -> dict[tuple[str, str, int], dict[str, object]]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    results: dict[tuple[str, str, int], dict[str, object]] = {}
    try:
        for key, audio in audio_by_sample.items():
            _voice, language, _repetition = key
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
            results[key] = {
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


def _aggregate_measurements(
    measurements: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for voice in dict.fromkeys(str(item["voice"]) for item in measurements):
        for language in dict.fromkeys(str(item["language"]) for item in measurements):
            items = [item for item in measurements if item["voice"] == voice and item["language"] == language]
            successful = [item for item in items if item["outcome"] == "success"]
            group: dict[str, object] = {
                "voice": voice,
                "language": language,
                "attempts": len(items),
                "successful": len(successful),
                "failed": len(items) - len(successful),
                "failure_codes": sorted(
                    {str(item["error_code"]) for item in items if item["outcome"] == "failure"}
                ),
                "thermal_admission_retries": sum(int(item["thermal_admission_retries"]) for item in items),
            }
            if successful:
                for field in (
                    "generation_seconds",
                    "total_worker_seconds",
                    "request_seconds",
                    "output_duration_seconds",
                    "real_time_factor",
                    "first_audio_latency_seconds",
                    "clipped_sample_percent",
                    "peak_absolute_amplitude",
                    "rms_amplitude",
                ):
                    values = [float(item[field]) for item in successful]
                    group[field] = {
                        "minimum": round(min(values), 6),
                        "median": round(median(values), 6),
                        "maximum": round(max(values), 6),
                    }
                group["maximum_clipped_sample_count"] = max(
                    int(item["clipped_sample_count"]) for item in successful
                )
                group["maximum_device_memory_used_mb"] = max(
                    float(item["device_memory"]["peak_used_mb"]) for item in successful
                )
                group["maximum_device_memory_delta_mb"] = max(
                    float(item["device_memory"]["peak_delta_mb"]) for item in successful
                )
                group["maximum_temperatures_c"] = {
                    sensor: max(float(item["peak_temperatures_c"][sensor]) for item in successful)
                    for sensor in ("gpu_edge_celsius", "cpu_package_celsius")
                }
                word_error_rates = [float(item["transcription"]["word_error_rate"]) for item in successful]
                group["word_error_rate"] = {
                    "minimum": round(min(word_error_rates), 6),
                    "median": round(median(word_error_rates), 6),
                    "maximum": round(max(word_error_rates), 6),
                }
                group["wav_validation_passed"] = all(
                    bool(item["wav_validation"]["valid"]) for item in successful
                )
            groups.append(group)
    return groups


def _error_code(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    code = payload.get("error", {}).get("code")
    return str(code) if code else f"http_{response.status_code}"


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
