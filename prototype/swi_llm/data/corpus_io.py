"""
swi_llm/data/corpus_io.py

コーパスの Parquet シャード入出力（roadmap/llm ステップ5 / fix 03）。

JSONL 単一ファイルは TB 級で行走査・並列読込・列圧縮のすべてが不利になる。
前処理済みコーパスを固定行数のシャード（shard-00000.parquet, ...）+ manifest.json で
保存し、下流（data_mix / 将来の分散読込）が順序安定・部分読込できるようにする。

- 依存は pyarrow のみ（HF datasets の既存依存。新規インストール不要）。
- 読み込みはシャード単位のバッチイテレータで、全量をメモリへ載せない。
"""
import json
import os

from swi_llm.utils.logger import setup_logger

logger = setup_logger("data.corpus_io")

MANIFEST_FILE = "manifest.json"
_SCHEMA_VERSION = "text-v1"
DEFAULT_SHARD_ROWS = 50_000


def write_parquet_shards(records, out_dir, shard_rows=DEFAULT_SHARD_ROWS):
    """レコード列（{"id":..., "text":...}）を Parquet シャードへ書き出す。

    Args:
        records: dict のイテラブル。text 必須・id 任意（欠損時は連番）。
        out_dir: 出力ディレクトリ（無ければ作成。既存 manifest は上書き）。
        shard_rows: 1 シャードあたりの行数。

    Returns:
        書き出した行数。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    shards = []
    ids, texts = [], []
    n_rows = 0

    def _flush():
        nonlocal ids, texts
        if not texts:
            return
        shard_name = f"shard-{len(shards):05d}.parquet"
        table = pa.table({"id": pa.array(ids, type=pa.int64()),
                          "text": pa.array(texts, type=pa.string())})
        pq.write_table(table, os.path.join(out_dir, shard_name))
        shards.append({"file": shard_name, "rows": len(texts)})
        ids, texts = [], []

    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue
        ids.append(int(rec.get("id", n_rows)))
        texts.append(text)
        n_rows += 1
        if len(texts) >= shard_rows:
            _flush()
    _flush()

    manifest = {"schema": _SCHEMA_VERSION, "n_rows": n_rows, "shards": shards}
    with open(os.path.join(out_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    logger.info(f"Parquet corpus written: {n_rows} rows in {len(shards)} shard(s) at {out_dir}")
    return n_rows


def has_parquet_corpus(dir_path):
    """有効な Parquet コーパス（manifest あり）かを判定する。"""
    return os.path.exists(os.path.join(dir_path or "", MANIFEST_FILE))


def iter_parquet_texts(dir_path, batch_rows=8192):
    """Parquet コーパスから text をシャード順・行順に yield する（全量非保持）。"""
    import pyarrow.parquet as pq

    manifest_path = os.path.join(dir_path, MANIFEST_FILE)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported corpus schema: {manifest.get('schema')!r} at {dir_path}")

    for shard in manifest["shards"]:
        pf = pq.ParquetFile(os.path.join(dir_path, shard["file"]))
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text
