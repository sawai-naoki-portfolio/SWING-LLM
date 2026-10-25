"""
swi_image — 画像生成AI（Text-to-Image）パッケージ。

姉妹パッケージ swi_llm / swi_voice と同じ prototype/ 直下に置かれ、SWI_MODE
(TEST/SMOKE/PROD/MAX) とロガー・Transformer 基盤を共有する。設計は
Doc/design.md（5. 画像生成 AI）を参照。

Phase 0（0→1）: 画像 → VAE 潜在空間 → DiT(Flow Matching) → Euler サンプラー(CFG) → 画像。
"""
__version__ = "0.1.0"
