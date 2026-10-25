"""
swi_voice/data/asr_pseudo_label.py
ASRを用いた疑似ラベリング (ASR Pseudo Labeling) [音声編1.2]

音声バッチを `asr_model.transcribe(audio) -> str` で書き起こし、既存の書き起こし
（無ければクリップに元々付いている `text`）との単語誤り率（WER）が閾値を超える
ペアを「テキストと音声が不一致」とみなして除外する。既存の書き起こしが無い
クリップは ASR 出力をそのまま疑似ラベルとして採用する。

`asr_model` は外部から注入する（Dependency Injection）。実運用では Whisper 等の
ASR モデルをこのインターフェイス（`.transcribe(audio) -> str`）でラップして渡す。
"""
from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import VOICE_ASR_WER_THRESHOLD

logger = setup_logger("voice.asr_pseudo_label")


def _tokenize_for_wer(text: str):
    """WER 計算用のトークン列。空白区切り言語は単語単位、空白の無いテキスト
    （日本語等の CJK）は文字単位にフォールバックする（実質 CER として機能する）。"""
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if len(words) > 1 or " " in text:
        return words
    return list(text)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """参照 `reference` に対する `hypothesis` の誤り率（編集距離 / 参照トークン数）。

    レーベンシュタイン距離（挿入・削除・置換）を動的計画法で計算する。参照が空の
    場合は、仮説も空なら 0.0、そうでなければ 1.0（全て誤り）を返す。
    """
    ref = _tokenize_for_wer(reference)
    hyp = _tokenize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,          # 削除
                curr[j - 1] + 1,      # 挿入
                prev[j - 1] + cost,   # 一致/置換
            )
        prev = curr
    return prev[m] / n


def pseudo_label_and_filter(audio_batch, asr_model, existing_transcripts=None,
                             wer_threshold=VOICE_ASR_WER_THRESHOLD):
    """音声バッチを ASR で書き起こし、WER 閾値でテキスト-音声の不整合ペアを除外する。

    Args:
        audio_batch: dict のリスト。各要素は最低限 `{"audio": ...}` を持つ
            （`swi_voice.data.audio_dataset` のクリップ形式 `{"audio","text",...}` と
            互換）。空/None の要素は安全にスキップする。
        asr_model: `.transcribe(audio) -> str` を実装するオブジェクト。None の場合は
            ASR 処理を実行できないため、警告を出して空リストを返す（silent success
            にせず、呼び出し側に「何も処理されなかった」ことが分かる形にする）。
        existing_transcripts: {id または batch内インデックス: text} の dict。
            指定が無い場合は各クリップの `text` フィールドを参照テキストとして使う。
        wer_threshold: 許容する最大 WER。これを超えるペアは除外する。

    Returns:
        フィルタ後のクリップ（`text` を確定した参照/疑似ラベルへ更新済み）のリスト。
        元の dict は変更せず、更新後のコピーを返す。
    """
    if not audio_batch:
        return []
    if asr_model is None:
        logger.warning("pseudo_label_and_filter: asr_model is None; cannot transcribe. Returning [].")
        return []

    results = []
    dropped_wer, dropped_bad_audio, pseudo_labeled = 0, 0, 0
    for idx, item in enumerate(audio_batch):
        if item is None:
            dropped_bad_audio += 1
            continue
        audio = item.get("audio") if hasattr(item, "get") else getattr(item, "audio", None)
        if audio is None or (hasattr(audio, "numel") and audio.numel() == 0):
            logger.warning(f"pseudo_label_and_filter: item {idx} has empty/missing audio; skipping.")
            dropped_bad_audio += 1
            continue

        try:
            asr_text = asr_model.transcribe(audio)
        except Exception:
            logger.exception(f"pseudo_label_and_filter: ASR transcription failed for item {idx}; skipping.")
            dropped_bad_audio += 1
            continue
        asr_text = (asr_text or "").strip()
        if not asr_text:
            logger.warning(f"pseudo_label_and_filter: ASR produced empty transcript for item {idx}; skipping.")
            dropped_bad_audio += 1
            continue

        item_id = item.get("id", idx) if hasattr(item, "get") else getattr(item, "id", idx)
        reference = None
        if existing_transcripts is not None:
            reference = existing_transcripts.get(item_id)
        if reference is None:
            reference = item.get("text") if hasattr(item, "get") else getattr(item, "text", None)

        if reference:
            wer = word_error_rate(asr_text, reference)
            if wer > wer_threshold:
                dropped_wer += 1
                continue
            final_text = reference
        else:
            pseudo_labeled += 1
            final_text = asr_text   # 書き起こしが無ければ ASR 出力を疑似ラベルとして採用

        merged = dict(item) if hasattr(item, "keys") else {"audio": audio}
        merged["text"] = final_text
        results.append(merged)

    logger.info(f"pseudo_label_and_filter: kept {len(results)}/{len(audio_batch)} "
                f"(dropped: wer={dropped_wer}, bad_audio={dropped_bad_audio}, "
                f"pseudo_labeled={pseudo_labeled}).")
    return results
