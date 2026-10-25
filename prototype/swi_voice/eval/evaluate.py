"""
swi_voice/eval/evaluate.py
評価スイート [音声編12.1, 実装編10.2]

「作ったら終わり」にせず、変更のたびに品質を客観的に測り続けるための評価軸を実装する。

  - 明瞭度 (WER): 生成音声を再び ASR に通し、元テキストとの単語誤り率を測る。
    値が小さいほど「意図したテキストがそのまま聞き取れる」。
  - 話者類似度 (SIM): 参照音声と生成音声の話者埋め込みのコサイン類似度。ゼロショット
    クローニングが声色を保持できているかを測る。値が大きいほど参照話者に近い。
  - 自然さ (MOS): eval/mos_predictor.MOSPredictor による自動 MOS 推定。

話者埋め込みは、学習済みコーデックのエンコーダ出力を時間方向に平均した latent を
プロキシとして用いる（依存ゼロ。実運用では ECAPA-TDNN 等の話者照合モデルへ差し替える）。
ASR は `.transcribe(audio) -> str` の DI インターフェイスで受け取る（data/asr_pseudo_label と同型）。

`VoiceEvalSuite.evaluate()` は集計値と各サンプルの内訳を返し、カスケード方式と
ネイティブ S2S 方式を同一タスクセットで比較する土台になる [音声編12.1]。
"""
import torch
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_voice.data.asr_pseudo_label import word_error_rate

logger = setup_logger("voice.evaluate")


def _to_batch(waveform: torch.Tensor) -> torch.Tensor:
    """波形を [B,1,T] へ整形する（[T] / [B,T] / [B,1,T] のいずれを渡しても対応）。

    dim==2 は [B,T] とみなして channel 次元を挿入する（B=1 なら [1,T]→[1,1,T] と同義）。
    """
    if waveform.dim() == 1:
        return waveform.view(1, 1, -1)
    if waveform.dim() == 2:
        return waveform.unsqueeze(1)
    return waveform


# 埋め込みノルムがこれ未満なら「話者情報を含まない」とみなす（無音など）。
_MIN_EMBED_NORM = 1e-4


@torch.no_grad()
def speaker_embedding(waveform, codec):
    """波形 → 話者埋め込み [B, dim]（コーデックの連続 latent の時間平均をプロキシとする）。

    量子化**前**の連続 latent (`encode_latent`) を使う。量子化後のコードは、コーデックが
    十分に学習されていないとコードブックが崩壊して全入力で同一になり、類似度が常に 1.0 に
    張り付いて弁別できなくなるため。

    さらに、エンコーダの定数バイアス（無音入力に対する応答）を差し引く。この DC 成分は
    全入力に共通で大きく、そのままコサイン類似度を取るとどの 2 音声でも ~1.0 に飽和する。
    同じ長さのゼロ信号を通して得たバイアスを引くことで、音色の差だけを残す。

    V-2: 空テンソル・短すぎる入力は想定内として WARNING を出し None を返す。
    """
    if waveform is None or waveform.numel() == 0:
        logger.warning("speaker_embedding: empty waveform; skipping.")
        return None
    wav = _to_batch(waveform)
    try:
        z = codec.encode_latent(wav)                             # [B, T', dim] 連続 latent
        bias = codec.encode_latent(torch.zeros_like(wav))        # 同じ長さの無音応答（DC 成分）
    except RuntimeError:
        # エンコーダの受容野に満たない極端に短い波形（想定内。呼び出し側でスキップさせる）
        logger.warning("speaker_embedding: waveform too short for the codec encoder; skipping.")
        return None
    if z.shape[1] == 0:
        logger.warning("speaker_embedding: codec produced 0 frames; skipping.")
        return None
    return z.mean(dim=1) - bias.mean(dim=1)                      # [B, dim] DC 除去済み


@torch.no_grad()
def speaker_similarity(ref_waveform, gen_waveform, codec):
    """参照音声と生成音声の話者類似度（コサイン類似度、[-1,1]）[音声編12.1]。

    クローニングの声色保持を測る。同一波形なら 1.0。どちらかが空・短すぎる、あるいは
    話者情報を持たない（無音＝埋め込みがほぼゼロベクトル）場合は None を返す
    （0.0 を返すと「似ていない」と誤読されるため、未定義であることを明示する）。
    """
    ref_emb = speaker_embedding(ref_waveform, codec)
    gen_emb = speaker_embedding(gen_waveform, codec)
    if ref_emb is None or gen_emb is None:
        return None
    if ref_emb.norm() < _MIN_EMBED_NORM or gen_emb.norm() < _MIN_EMBED_NORM:
        logger.warning("speaker_similarity: near-zero embedding (silent audio?); similarity undefined.")
        return None
    return float(F.cosine_similarity(ref_emb, gen_emb, dim=-1).mean())


def transcription_wer(gen_waveform, expected_text, asr_model):
    """生成音声を再 ASR し、意図テキストとの WER を返す（明瞭度）[音声編12.1]。

    ASR 失敗時は例外を伝播させず WARNING を出して None を返す（1 件の失敗で評価全体を
    落とさない）。`asr_model` が None の場合も None（WER 未計測）。
    """
    if asr_model is None:
        return None
    if gen_waveform is None or gen_waveform.numel() == 0:
        logger.warning("transcription_wer: empty generated waveform; skipping.")
        return None
    try:
        hypothesis = asr_model.transcribe(gen_waveform)
    except Exception:
        logger.exception("transcription_wer: ASR transcription failed; skipping this sample.")
        return None
    return word_error_rate(hypothesis or "", expected_text or "")


def _mean(values):
    """None を除いた平均。有効値が無ければ None。"""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class VoiceEvalSuite:
    """WER / 話者類似度 / MOS をまとめて測る評価スイート [音声編12.1]。

    Args:
        codec: 話者埋め込みの抽出に使う NeuralAudioCodec（None なら SIM を測らない）。
        asr_model: `.transcribe(audio) -> str`（None なら WER を測らない）。
        mos_predictor: `MOSPredictor`（None なら MOS を測らない）。

    いずれも省略可能で、注入されたものだけを測る（ASR が無い環境でも MOS/SIM は取れる）。
    """

    def __init__(self, codec=None, asr_model=None, mos_predictor=None):
        self.codec = codec
        self.asr_model = asr_model
        self.mos_predictor = mos_predictor
        logger.info(f"Initializing VoiceEvalSuite (wer={'on' if asr_model else 'off'}, "
                    f"sim={'on' if codec else 'off'}, mos={'on' if mos_predictor else 'off'}).")

    @torch.no_grad()
    def evaluate_one(self, gen_waveform, expected_text=None, ref_waveform=None):
        """1 サンプルの指標 dict {"wer","sim","mos"} を返す（測れない軸は None）。"""
        wer = None
        if self.asr_model is not None and expected_text is not None:
            wer = transcription_wer(gen_waveform, expected_text, self.asr_model)

        sim = None
        if self.codec is not None and ref_waveform is not None:
            sim = speaker_similarity(ref_waveform, gen_waveform, self.codec)

        mos = None
        if self.mos_predictor is not None:
            if gen_waveform is None or gen_waveform.numel() == 0:
                logger.warning("evaluate_one: empty generated waveform; skipping MOS.")
            else:
                mos = float(self.mos_predictor(_to_batch(gen_waveform)).mean())

        return {"wer": wer, "sim": sim, "mos": mos}

    @torch.no_grad()
    def evaluate(self, samples):
        """複数サンプルを評価する [音声編12.1]。

        Args:
            samples: dict のリスト。各要素は `{"audio": Tensor, "text": str, "ref_audio": Tensor}`。
                `text` / `ref_audio` は任意（無ければ該当指標をスキップ）。

        Returns:
            {"n": 件数, "wer": 平均, "sim": 平均, "mos": 平均, "per_sample": [各内訳]}。
            平均は測れた値のみで取り、1 件も測れなければ None。
        """
        if not samples:
            logger.warning("VoiceEvalSuite.evaluate: no samples provided.")
            return {"n": 0, "wer": None, "sim": None, "mos": None, "per_sample": []}

        per_sample = [
            self.evaluate_one(s.get("audio"), s.get("text"), s.get("ref_audio"))
            for s in samples
        ]
        result = {
            "n": len(per_sample),
            "wer": _mean([r["wer"] for r in per_sample]),
            "sim": _mean([r["sim"] for r in per_sample]),
            "mos": _mean([r["mos"] for r in per_sample]),
            "per_sample": per_sample,
        }
        logger.info(f"VoiceEvalSuite: n={result['n']} wer={result['wer']} "
                    f"sim={result['sim']} mos={result['mos']}")
        return result
