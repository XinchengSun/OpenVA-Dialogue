from __future__ import annotations

import unittest

from scripts.transcribe_reference import SENSEVOICE_LANGUAGES


class ReferenceLanguageTests(unittest.TestCase):
    def test_public_language_codes_map_to_sensevoice_hints(self):
        self.assertEqual(SENSEVOICE_LANGUAGES["auto"], "auto")
        self.assertEqual(SENSEVOICE_LANGUAGES["zh-CN"], "zh")
        self.assertEqual(SENSEVOICE_LANGUAGES["en-US"], "en")
        self.assertEqual(SENSEVOICE_LANGUAGES["ja-JP"], "ja")


if __name__ == "__main__":
    unittest.main()
