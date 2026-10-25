"""
画像生成推論 (swi_image/serve/generate.py) と WebUI 統合の単体テスト。
チェックポイント/トークナイザ未整備でも壊れないよう skip ガードする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import TOKENIZER_MODEL_PATH

_PROTO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOK = os.path.join(_PROTO, TOKENIZER_MODEL_PATH)


class TestImageServeHelpers(unittest.TestCase):
    def test_suffix_priority_includes_fallbacks(self):
        from swi_image.serve.generate import _suffix_priority
        order = _suffix_priority()
        self.assertIn("", order)
        self.assertIn("_TEST", order)
        self.assertGreaterEqual(len(order), 4)

    def test_image_available_returns_bool(self):
        from swi_image.serve.generate import image_available
        self.assertIsInstance(image_available(), bool)


@unittest.skipUnless(os.path.exists(_TOK), "tokenizer.model not trained yet")
class TestImageGeneration(unittest.TestCase):
    def test_generate_produces_valid_png(self):
        from swi_image.serve.generate import image_available, get_generator
        if not image_available():
            self.skipTest("image checkpoints not present (run swi_image training first)")
        import io
        from PIL import Image
        png = get_generator().generate("赤い円が中央にある青い背景", steps=4)
        self.assertEqual(png[:4], b"\x89PNG")            # 有効な PNG シグネチャ
        im = Image.open(io.BytesIO(png))
        self.assertEqual(im.mode, "RGB")
        self.assertGreater(im.size[0], 0)


if __name__ == "__main__":
    unittest.main()
