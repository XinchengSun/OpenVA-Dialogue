import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from latency_trace import trace, text_fields, media_fields


class LatencyTraceTests(unittest.TestCase):
    def test_disabled_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENVA_LATENCY_TRACE_DIR": ""}):
            trace("disabled")
            self.assertEqual(text_fields("private text"), {})
            self.assertEqual(media_fields(b"private media"), {})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_media_fingerprint_can_match_transport_without_recording_payload(self):
        import hashlib
        payload = b"private media"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENVA_LATENCY_TRACE_DIR": directory}):
            fields = media_fields(payload)
            trace("media_sent", **fields)
            record = json.loads(next(Path(directory).glob("*.jsonl")).read_text())
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["media_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertNotIn("private media", json.dumps(record))

    def test_threaded_records_are_complete_and_exclude_text(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENVA_LATENCY_TRACE_DIR": directory}):
            workers = [threading.Thread(target=lambda: [trace("event", **text_fields("private text")) for _ in range(20)]) for _ in range(3)]
            for worker in workers: worker.start()
            for worker in workers: worker.join()
            path = next(Path(directory).glob("*.jsonl"))
            raw = path.read_text()
            self.assertNotIn("private text", raw)
            rows = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len(rows), 60)
            self.assertTrue(all(row["characters"] == 12 and row["monotonic_ns"] > 0 for row in rows))

    def test_unwritable_destination_does_not_interrupt_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("existing")
            with patch.dict(os.environ, {"OPENVA_LATENCY_TRACE_DIR": str(path)}):
                trace("ignored_output_failure")
            self.assertEqual(path.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
