"""
swi_image/data/safety_filter.py
学習データの画像安全フィルタ [画像編1.3, NEW-I1 / 02_architecture_docs_gap_audit]

設計（02_Image_Generation_Architecture §1）は「安全性フィルタリング（最重要）」として
CSAM 検出・NSFW 分類・著作権配慮を要求する。従来は `IMG_NSFW_THRESHOLD` 定数だけの
「枠組みのみ」で分類器の実体が無かった。本モジュールで NSFW 分類を実体化する。

3 層の役割分担:
  1. **NSFW 分類（本モジュールで実体化）**: HF の画像分類モデル
     （既定 `Falconsai/nsfw_image_detection`, ViT ~350MB, 初回のみ DL・以後キャッシュ）で
     nsfw スコアを算出し、`IMG_NSFW_THRESHOLD` 以上を学習データから除外する。
  2. **CSAM ハッシュ照合（差し込み口のみ）**: PhotoDNA / NCMEC 等の公式ハッシュ DB は
     認可された組織にしか提供されず、個人開発環境では実装できない。`hash_checker` DI
     （`callable(PIL.Image) -> bool`、True = 照合ヒットで除外）を用意した。
     **公開データでの本格学習（image roadmap step4 以降）や公開サービス提供の前に、
     組織として照合体制を整備すること**。
  3. **著作権/ライセンス**: ソース単位で管理する（`IMG_DATASET_CANDIDATES` は公開
     ライセンスのキュレーション済みデータセットのみを列挙する、が現行の統制）。

縮退の方針: 分類器が使えない環境（未導入/オフライン）ではフィルタなしで続行するが、
**ERROR レベルで明示**する。安全フィルタの不在は「静かな縮退」（C-4 の WARNING）扱いに
しない。意図的に無効化する場合のみ `SWI_IMG_SAFETY=0` を設定する（WARNING）。
"""
import os
import threading

import torch

from swi_llm.utils.logger import setup_logger
from swi_image.utils.constants import (
    IMG_NSFW_THRESHOLD, IMG_SAFETY_ENV, IMG_NSFW_MODEL_ENV, IMG_NSFW_DEFAULT_MODEL,
)

logger = setup_logger("image.safety")


def safety_enabled() -> bool:
    """SWI_IMG_SAFETY=0 で明示的に無効化できる（既定は有効）。"""
    return os.environ.get(IMG_SAFETY_ENV, "1").strip().lower() not in ("0", "false", "no")


class NSFWClassifier:
    """HF image-classification パイプラインで nsfw スコアを返す（遅延ロード）。"""

    def __init__(self, model_name=None, device=None):
        self.model_name = (model_name or os.environ.get(IMG_NSFW_MODEL_ENV, "").strip()
                           or IMG_NSFW_DEFAULT_MODEL)
        self._device = device
        self._pipe = None
        self._lock = threading.Lock()

    def load(self):
        if self._pipe is not None:
            return self
        with self._lock:
            if self._pipe is not None:
                return self
            from transformers import pipeline
            device = self._device
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading NSFW classifier '{self.model_name}' (device={device})...")
            self._pipe = pipeline("image-classification", model=self.model_name, device=device)
        return self

    def nsfw_score(self, pil_image) -> float:
        """画像 1 枚の nsfw スコア [0,1] を返す。"""
        self.load()
        results = self._pipe(pil_image)
        for entry in results:
            if str(entry.get("label", "")).lower() == "nsfw":
                return float(entry.get("score", 0.0))
        return 0.0


class ImageSafetyFilter:
    """NSFW 分類 + CSAM ハッシュ照合（DI）を束ねる学習データフィルタ。

    Args:
        classifier: `nsfw_score(pil) -> float` を持つオブジェクト（既定 NSFWClassifier）。
        hash_checker: `callable(pil) -> bool`（True = CSAM 等の照合ヒット → 除外）。
            公式ハッシュ DB を持つ組織環境でのみ注入できる。既定 None。
        threshold: nsfw スコアの除外閾値。
    """

    def __init__(self, classifier=None, hash_checker=None, threshold=IMG_NSFW_THRESHOLD):
        self.classifier = classifier
        self.hash_checker = hash_checker
        self.threshold = threshold
        self.stats = {"checked": 0, "nsfw_dropped": 0, "hash_dropped": 0, "errors": 0}
        self._classifier_broken = False

    def is_safe(self, pil_image) -> bool:
        """学習データとして安全なら True。除外理由は stats に集計する。

        分類中の予期しない例外は 1 回だけ ERROR で報告し、以後の分類チェックを止めて
        フィルタなし相当で続行する（データロード全体を分類器のバグで落とさない。
        ただし不在は ERROR で可視化されているので「静かな」縮退にはならない）。
        """
        self.stats["checked"] += 1

        if self.hash_checker is not None:
            try:
                if self.hash_checker(pil_image):
                    self.stats["hash_dropped"] += 1
                    return False
            except Exception:
                logger.exception("CSAM hash checker failed; treating this image as UNSAFE (fail-closed).")
                self.stats["hash_dropped"] += 1
                return False

        if self.classifier is not None and not self._classifier_broken:
            try:
                score = self.classifier.nsfw_score(pil_image)
            except Exception:
                self.stats["errors"] += 1
                self._classifier_broken = True
                logger.exception(
                    "NSFW 分類器の実行に失敗しました。以後この読み込みでは NSFW フィルタなしで続行します"
                    "（安全フィルタ不在の状態です。NEW-I1 / 画像編1.3）。")
                return True
            if score >= self.threshold:
                self.stats["nsfw_dropped"] += 1
                logger.debug(f"NSFW filtered: score={score:.3f} >= {self.threshold}")
                return False
        return True

    def log_stats(self, context=""):
        s = self.stats
        if s["checked"]:
            logger.info(f"Image safety filter{f' ({context})' if context else ''}: "
                        f"checked={s['checked']} nsfw_dropped={s['nsfw_dropped']} "
                        f"hash_dropped={s['hash_dropped']} errors={s['errors']} "
                        f"(threshold={self.threshold})")


_filter_singleton = None
_filter_failed = False
_filter_lock = threading.Lock()


def get_safety_filter():
    """ImageSafetyFilter のシングルトン。無効化時・分類器が用意できない環境では None。

    None の意味は 2 通りあり、ログレベルで区別する:
      - SWI_IMG_SAFETY=0 の明示無効化 → WARNING（運用者の意思）
      - 分類器が使えない（transformers 不在 / 重み取得不可）→ **ERROR**（安全フィルタ不在で
        学習データを取り込む状態。設計文書が「最重要」とする統制の欠落なので目立たせる）
    """
    global _filter_singleton, _filter_failed
    if not safety_enabled():
        logger.warning(f"画像の学習データ安全フィルタが明示的に無効化されています（{IMG_SAFETY_ENV}=0）。")
        return None
    if _filter_singleton is not None:
        return _filter_singleton
    if _filter_failed:
        return None
    with _filter_lock:
        if _filter_singleton is not None or _filter_failed:
            return _filter_singleton
        try:
            classifier = NSFWClassifier()
            classifier.load()          # ここでロードを確定させる（不可なら except へ）
            _filter_singleton = ImageSafetyFilter(classifier=classifier)
            logger.info("Image safety filter enabled (NSFW classifier loaded, NEW-I1).")
        except Exception as e:
            _filter_failed = True
            logger.error(
                f"NSFW 分類器を用意できませんでした（{type(e).__name__}: {e}）。"
                "安全フィルタなしで学習データを取り込みます。オンライン環境で再実行して重みを取得するか、"
                f"意図的に無効化する場合は {IMG_SAFETY_ENV}=0 を設定してください（画像編1.3 / NEW-I1）。")
    return _filter_singleton


def reset_safety_filter():
    """シングルトン状態を初期化する（テスト用）。"""
    global _filter_singleton, _filter_failed
    with _filter_lock:
        _filter_singleton = None
        _filter_failed = False
