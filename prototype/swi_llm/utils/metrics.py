"""
swi_llm/utils/metrics.py
学習メトリクスの JSONL ロガー [NEW-Q2 / 04_Quality_and_Standards §1 S-3]

設計は「学習中の Loss 推移、GPU メモリ使用量、スループット等のメトリクス監視
（W&B 等の連携想定）」を要求するが、従来は logger のテキスト出力しかなく、
機械可読な時系列が存在しなかった。

方式（ローカルファースト）:
  - 1 行 = 1 レコードの JSONL を `errorLog/metrics/<run>_<MODE>.jsonl` へ追記する。
    （run.ps1 のログクリアは *.log のみ対象なので、メトリクス履歴は run を跨いで残る。
    レコードに run_id(起動時刻) を含み、後から run 単位に分離できる。）
  - 記録: 時刻 / epoch / step / loss / lr / GPU メモリ / スループット(samples/sec)。
  - ホットパス配慮（REQ-10）: コンソール logger には出さない。ファイル append のみ。
  - W&B / TensorBoard は「この JSONL を読み込む」形で連携する（アダプタ側の責務。
    ロガー本体を外部サービスへ依存させない）。
  - SWI_METRICS=0 で無効化（既定は有効）。

使い方:
    metrics = MetricsLogger("sft")                 # 無効時は全メソッドが no-op
    metrics.log_step(epoch=0, step=10, loss=2.31, lr=3e-4, n_samples=16)
    metrics.log_epoch(epoch=0, loss=2.10)
    metrics.close()
"""
import itertools
import json
import os
import time

import torch

METRICS_DISABLE_ENV = "SWI_METRICS"
# swi_llm/utils/metrics.py → swi_llm → prototype
_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_METRICS_DIR = os.path.join(_PROTOTYPE_DIR, "errorLog", "metrics")


def metrics_enabled() -> bool:
    return os.environ.get(METRICS_DISABLE_ENV, "1").strip().lower() not in ("0", "false", "no")


# プロセス内の run 連番。時刻だけだと同一ミリ秒内の連続生成で run_id が衝突する
# （Windows の時計分解能でテストが実証）。プロセスを跨ぐ分は時刻部分が区別する。
_run_counter = itertools.count()


class MetricsLogger:
    """学習ランのメトリクスを JSONL に追記する軽量ロガー。

    Args:
        run_name: trainer 名（"pretrain", "sft", "voice_codec" 等）。ファイル名になる。
        mode: SWI_MODE（省略時は環境変数から）。
        out_dir: 出力ディレクトリ（省略時 errorLog/metrics）。
        enabled: 省略時は SWI_METRICS に従う。False なら全メソッド no-op。
    """

    def __init__(self, run_name, mode=None, out_dir=None, enabled=None):
        self.enabled = metrics_enabled() if enabled is None else enabled
        if not self.enabled:
            self._fh = None
            return
        mode = mode or os.environ.get("SWI_MODE", "TEST").strip().upper() or "TEST"
        self.run_name = run_name
        self.mode = mode
        # 時刻 + プロセス内連番（同一プロセスで連続生成しても決定的に一意）。
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + f".{next(_run_counter):03d}"
        out_dir = out_dir or DEFAULT_METRICS_DIR
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{run_name}_{mode}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._last_step_time = None

    def _gpu_mem_mb(self):
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        return None

    def _write(self, record):
        record = {"ts": round(time.time(), 3), "run_id": self.run_id,
                  "run": self.run_name, "mode": self.mode, **record}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log_step(self, epoch, step, loss, lr=None, n_samples=None, **extra):
        """optimizer ステップ単位の記録。n_samples を渡すと samples/sec を算出する。"""
        if not self.enabled:
            return
        # perf_counter を使う（Windows の time.monotonic は分解能 ~15.6ms で、
        # 速いステップ間の経過時間が 0 になりスループットが測れない — 実測で発覚）。
        now = time.perf_counter()
        throughput = None
        if n_samples is not None and self._last_step_time is not None:
            dt = now - self._last_step_time
            if dt > 0:
                throughput = round(n_samples / dt, 2)
        self._last_step_time = now
        self._write({"kind": "step", "epoch": epoch, "step": step,
                     "loss": None if loss is None else round(float(loss), 6),
                     "lr": None if lr is None else float(lr),
                     "gpu_mem_mb": self._gpu_mem_mb(),
                     "samples_per_sec": throughput, **extra})

    def log_epoch(self, epoch, loss=None, **extra):
        """エポック単位の記録（voice / image の epoch 平均損失など）。"""
        if not self.enabled:
            return
        self._write({"kind": "epoch", "epoch": epoch,
                     "loss": None if loss is None else round(float(loss), 6),
                     "gpu_mem_mb": self._gpu_mem_mb(), **extra})

    def log_eval(self, **fields):
        """評価結果の記録（kind=eval。perplexity 等の成長の推移を run を跨いで追う）。"""
        if not self.enabled:
            return
        self._write({"kind": "eval", **fields})

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_shared = {}


def log_epoch_metric(run_name, epoch, loss=None, **extra):
    """エポック単位メトリクスの 1 行呼び出し口（voice / image trainer 用）。

    run_name ごとに MetricsLogger をプロセス内でキャッシュする（書き込みは都度 flush
    されるため、スクリプト終了時に close しなくても取りこぼさない）。
    """
    lg = _shared.get(run_name)
    if lg is None:
        lg = _shared[run_name] = MetricsLogger(run_name)
    lg.log_epoch(epoch=epoch, loss=loss, **extra)
