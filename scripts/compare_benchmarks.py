#!/usr/bin/env python3
"""Compare two single-GPU reports without hiding failed or unmatched runs.

Usage: python scripts/compare_benchmarks.py baseline/report.json candidate/report.json
Exit 2 means known comparison gaps need review; metric deltas remain descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def value_at(report: dict, *path):
    for key in path:
        if not isinstance(report, dict):
            return None
        report = report.get(key)
    return report


def delta(before, after) -> dict:
    valid = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    result = {"baseline": before if valid(before) else None, "candidate": after if valid(after) else None,
              "absolute_delta": None, "relative_delta_pct": None}
    if valid(before) and valid(after):
        result["absolute_delta"] = after - before
        if before != 0:
            result["relative_delta_pct"] = (after - before) / abs(before) * 100
    return result


def target_devices(report: dict) -> list:
    samples = value_at(report, "capture", "samples") or []
    devices = set()
    for sample in samples:
        target = value_at(sample, "gpu", "target")
        if (not isinstance(target, dict) or not isinstance(target.get("uuid"), str) or not target["uuid"]
                or not isinstance(target.get("name"), str) or not target["name"]
                or not isinstance(target.get("memory_total_mib"), (int, float))
                or isinstance(target["memory_total_mib"], bool)
                or not math.isfinite(target["memory_total_mib"]) or target["memory_total_mib"] <= 0):
            return []
        devices.add((str(target["uuid"]), str(target["name"]), target["memory_total_mib"]))
    return sorted(devices)


def compare(before: dict, after: dict) -> dict:
    issues = []
    for label, report in (("baseline", before), ("candidate", after)):
        if report.get("status") != "pass":
            issues.append(f"{label}: benchmark status is not pass")
        provenance = report.get("provenance") or {}
        if not provenance:
            issues.append(f"{label}: provenance is missing")
        checkout = provenance.get("checkout") or {}
        if not checkout.get("commit") or checkout.get("dirty_tracked") is not False or checkout.get("untracked_file_count") != 0:
            issues.append(f"{label}: benchmark checkout is not a recorded clean commit")
        if provenance.get("profile") in (None, "unspecified"):
            issues.append(f"{label}: runtime profile is unspecified")
        if not provenance.get("runtime_config_sha256"):
            issues.append(f"{label}: no runtime configuration fingerprint supplied")
        if not provenance.get("declared_service_commit"):
            issues.append(f"{label}: no operator-verified service commit declared")
        for field in ("declared_weights_manifest_sha256", "measurement", "measurement_runtime"):
            if not provenance.get(field):
                issues.append(f"{label}: required provenance is missing: {field}")
        measurement = provenance.get("measurement")
        if (not isinstance(measurement, dict) or not all(
                key in measurement for key in ("duration_sec", "sample_sec", "input_delay_sec", "turns", "turn_timeout_sec", "voice_threshold"))):
            issues.append(f"{label}: measurement settings are incomplete")
        runtime = provenance.get("measurement_runtime")
        if (not isinstance(runtime, dict) or not all(
                key in runtime for key in ("python", "system", "release", "machine", "packages"))):
            issues.append(f"{label}: measurement runtime is incomplete")
        for path in (("decoded_frames_delivered_per_wall_second",), ("total_target_gpu_memory_mib", "max")):
            value = value_at(report, *path)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                issues.append(f"{label}: invalid or missing metric: {'.'.join(path)}")
    for field in ("profile", "declared_weights_manifest_sha256", "runtime_config_sha256",
                  "input_wav_sha256", "instruction_sha256", "measurement", "measurement_runtime"):
        if value_at(before, "provenance", field) != value_at(after, "provenance", field):
            issues.append(f"mismatched provenance: {field}")
    before_devices, after_devices = target_devices(before), target_devices(after)
    if not before_devices or before_devices != after_devices or len(before_devices) != 1:
        issues.append("target GPU identity is missing, changed, or inconsistent")
    metrics = {
        "delivered_fps": delta(before.get("decoded_frames_delivered_per_wall_second"), after.get("decoded_frames_delivered_per_wall_second")),
        "sampled_peak_gpu_memory_mib": delta(value_at(before, "total_target_gpu_memory_mib", "max"), value_at(after, "total_target_gpu_memory_mib", "max")),
    }
    for field in ("latest_frame_age_sec", "latest_visible_frame_age_sec"):
        metrics[field + "_p95"] = delta(value_at(before, "frame_freshness", field, "p95"), value_at(after, "frame_freshness", field, "p95"))
    for field in ("count", "p50", "p95"):
        metrics["server_voice_end_to_media_boundary_ms_" + field] = delta(
            value_at(before, "dialogue_latency_ms", "assistant_media_boundary", field),
            value_at(after, "dialogue_latency_ms", "assistant_media_boundary", field),
        )
    return {
        "status": "review_required" if issues else "matched_recorded_conditions",
        "comparison_gaps": issues,
        "baseline_measurement_commit": value_at(before, "provenance", "checkout", "commit"),
        "candidate_measurement_commit": value_at(after, "provenance", "checkout", "commit"),
        "baseline_declared_service_commit": value_at(before, "provenance", "declared_service_commit"),
        "candidate_declared_service_commit": value_at(after, "provenance", "declared_service_commit"),
        "metrics": metrics,
        "limitations": [
            "Deltas are descriptive, not evidence of statistical significance or perceptual quality.",
            "Matched recorded conditions do not attest the running service revision, loaded weights, warm-up, or external provider load.",
            "Delivered FPS includes held frames and is not model-generation FPS; browser playback is outside this benchmark.",
            "If configuration intentionally changed, review the mismatch and retain both reports; do not silently waive it.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text(encoding="utf-8")), json.loads(args.candidate.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if result["comparison_gaps"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
