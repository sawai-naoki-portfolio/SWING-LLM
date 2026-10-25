"""
swi_voice — 音声合成AI（TTS / Voice AI）パッケージ。

姉妹パッケージ swi_llm と同じ prototype/ 直下に置かれ、SWI_MODE
(TEST/SMOKE/PROD/MAX) とロガー・Transformer 基盤を共有する。
設計は Doc/design.md（4. 音声 AI）を参照。

Phase 0（0→1）: 音声データ → ニューラルコーデック(RVQ) → TTS(VALL-E系AR) → 波形。
"""
__version__ = "0.1.0"
