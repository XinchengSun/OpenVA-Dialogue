import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "setup_weights.py"
SPEC = importlib.util.spec_from_file_location("setup_weights", SCRIPT_PATH)
assert SPEC and SPEC.loader
setup_weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_weights)


class WeightsManifestTests(unittest.TestCase):
    def test_repository_manifest_is_valid_and_never_guesses_hashes(self):
        manifest = setup_weights.load_manifest(REPO_ROOT / "weights-manifest.json")
        ids = {model["id"] for model in manifest["models"]}
        self.assertEqual(
            ids,
            {
                "dystream-motion",
                "dystream-lia-renderer",
                "wav2vec2-base-960h",
                "sensevoice-small",
                "fish-audio-s2-pro",
            },
        )
        dy_models = [model for model in manifest["models"] if model["id"].startswith("dystream-")]
        self.assertTrue(all(model["redistribution"] == "upstream-only" for model in dy_models))
        self.assertTrue(all("unknown" in model["license"] for model in dy_models))
        fish = next(model for model in manifest["models"] if model["id"] == "fish-audio-s2-pro")
        self.assertTrue(fish["required"])
        self.assertEqual(fish["license"], "Fish Audio Research License")
        self.assertEqual(fish["role"], "primary-tts")
        self.assertEqual(
            fish["revision"],
            "1de9996b6be38b745688de084d87a5633f714e4e",
        )
        self.assertEqual(
            fish["directory_target"],
            "runtime/fish-s2-pro/models/fishaudio-s2-pro",
        )
        self.assertIn("codec.pth", fish["required_files"])

    def test_verify_checks_size_and_known_hash_but_allows_unknown_hash(self):
        payload = b"verified fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "models" / "fixture.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            model = {
                "id": "fixture",
                "files": [
                    {
                        "target": "models/fixture.bin",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            self.assertEqual(setup_weights.verify_model(root, model, deep=True), [])
            model["files"][0]["sha256"] = None
            self.assertEqual(setup_weights.verify_model(root, model, deep=True), [])
            model["files"][0]["size_bytes"] += 1
            self.assertIn("size mismatch", setup_weights.verify_model(root, model, deep=False)[0])

    def test_directory_model_must_be_nonempty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = {
                "id": "directory",
                "files": [],
                "directory_target": "models/directory",
                "required_files": ["config.json"],
            }
            target = root / model["directory_target"]
            target.mkdir(parents=True)
            self.assertIn("empty model directory", setup_weights.verify_model(root, model, deep=True)[0])
            (target / "unrelated.txt").write_text("x", encoding="utf-8")
            self.assertIn("missing required file", setup_weights.verify_model(root, model, deep=True)[0])
            (target / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(setup_weights.verify_model(root, model, deep=True), [])

    def test_target_cannot_escape_install_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(setup_weights.ManifestError):
                setup_weights.safe_target(root, "../outside.bin")
            with self.assertRaises(setup_weights.ManifestError):
                setup_weights.safe_target(root, str((root.parent / "outside.bin").resolve()))

    def test_unknown_model_selection_is_rejected(self):
        manifest = {"models": [{"id": "known", "required": True}]}
        with self.assertRaises(setup_weights.ManifestError):
            setup_weights.selected_models(manifest, ["unknown"], include_optional=False)

    def test_download_does_not_silently_accept_an_invalid_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "weights" / "model.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"bad")
            model = {
                "id": "existing",
                "provider": "huggingface",
                "files": [
                    {
                        "target": "weights/model.bin",
                        "size_bytes": 100,
                        "sha256": None,
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "rerun with --force"):
                setup_weights.download_model(root, model, force=False)


if __name__ == "__main__":
    unittest.main()
