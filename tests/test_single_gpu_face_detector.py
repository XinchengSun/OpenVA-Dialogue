"""Exercise the actual preprocessing loader without importing the model stack."""
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class FaceDetectorDeviceTests(unittest.TestCase):
    def test_single_gpu_uses_cpu_delegate_without_changing_legacy_default(self):
        source = Path(__file__).resolve().parents[1] / "app.py"
        function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "load_face_detector")
        for setting, delegate in (("", 1), ("0", 0), ("6", 0)):
            with self.subTest(setting=setting):
                factory = Mock()
                egl_seen = []
                factory.side_effect = lambda **_: egl_seen.append(
                    os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")) or object()
                imports = SimpleNamespace(
                    spec_from_file_location=Mock(return_value=SimpleNamespace(
                        loader=SimpleNamespace(exec_module=Mock()))),
                    module_from_spec=Mock(return_value=SimpleNamespace(FaceDetector=factory)),
                )
                scope = dict(os=os, _ilu=imports, VIS_DIR="fixture", _face_detector=None)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
                with patch.dict(os.environ, {"DYSTREAM_SINGLE_GPU": setting,
                                            "__EGL_VENDOR_LIBRARY_FILENAMES": "existing.json"}), \
                        patch("os.path.exists", return_value=True):
                    scope["load_face_detector"]()
                    scope["load_face_detector"]()
                    self.assertEqual(os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"], "existing.json")
                self.assertEqual(egl_seen, ["" if setting else "existing.json"])
                factory.assert_called_once_with(
                    mediapipe_model_asset_path=os.path.join("fixture", "utils", "face_landmarker.task"),
                    face_detection_confidence=0.5, num_faces=1, delegate=delegate)


if __name__ == "__main__":
    unittest.main()
