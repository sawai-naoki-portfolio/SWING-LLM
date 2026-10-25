"""
Phase 2: パープレキシティ評価（swi_llm/eval/perplexity.py）の検証。

計算そのものは「答えが分かっている」偽モデルで確かめる:
  一様な出力 → PPL = 語彙数、正解を確信する出力 → PPL ≈ 1。
"""
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.eval import perplexity
from swi_llm.eval.perplexity import compute_perplexity, heldout_tokens
from swi_llm.utils import metrics as metrics_mod

V = 50


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))     # evaluate() が device を取れるように

    def forward(self, x):
        return torch.zeros(*x.shape, V), torch.tensor(0.0)


class OracleModel(torch.nn.Module):
    """系列が 0,1,2,... と 1 ずつ増えると知っているモデル（次は x+1 と確信する）。"""

    def forward(self, x):
        logits = torch.full((*x.shape, V), -1e4)
        logits.scatter_(-1, ((x + 1) % V).unsqueeze(-1), 0.0)
        return logits, torch.tensor(0.0)


class TestComputePerplexity(unittest.TestCase):
    def test_uniform_model_has_vocab_size_perplexity(self):
        out = compute_perplexity(UniformModel(), np.arange(100) % V, seq_len=16)
        self.assertAlmostEqual(out["ppl"], V, places=3)
        self.assertAlmostEqual(out["nll"], math.log(V), places=5)

    def test_oracle_model_has_perplexity_one(self):
        out = compute_perplexity(OracleModel(), np.arange(100) % V, seq_len=16)
        self.assertAlmostEqual(out["ppl"], 1.0, places=3)

    def test_counts_every_next_token_including_the_tail_window(self):
        out = compute_perplexity(UniformModel(), np.arange(37) % V, seq_len=16, batch_size=2)
        self.assertEqual(out["tokens"], 36)                 # 37 トークン → 36 個の次トークン予測

    def test_max_tokens_limits_the_evaluated_span(self):
        out = compute_perplexity(UniformModel(), np.arange(1000) % V, seq_len=16, max_tokens=64)
        self.assertEqual(out["tokens"], 64)

    def test_mixed_quality_is_token_weighted(self):
        """半分を確信・半分を一様に当てるなら、平均 NLL は log(V)/2 付近（窓単位でなくトークン単位の平均）。"""
        class Half(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(*x.shape, V)
                good = (x % 2 == 0)
                oracle = torch.full((*x.shape, V), -1e4).scatter_(-1, ((x + 1) % V).unsqueeze(-1), 0.0)
                return torch.where(good.unsqueeze(-1), oracle, logits), torch.tensor(0.0)
        out = compute_perplexity(Half(), np.arange(400) % V, seq_len=32)
        self.assertAlmostEqual(out["nll"], math.log(V) / 2, delta=0.05)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            compute_perplexity(UniformModel(), np.array([3]), seq_len=16)


class TestHeldoutText(unittest.TestCase):
    def test_paragraphs_are_tokenized_and_eot_separated(self):
        from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID
        toks = heldout_tokens(get_tokenizer(), EOT_TOKEN_ID)
        with open(perplexity.HELDOUT_PATH, encoding="utf-8") as f:
            n_paragraphs = len([p for p in f.read().split("\n\n") if p.strip()])
        self.assertGreaterEqual(n_paragraphs, 10)
        self.assertEqual(int((toks == EOT_TOKEN_ID).sum()), n_paragraphs)
        self.assertGreater(len(toks), 2048)                 # seq_len 512 の窓が 4 つ以上取れる量


class TestCli(unittest.TestCase):
    def _run(self, result=None, error=None, argv=()):
        def fake_eval(*a, **kw):
            if error:
                raise error
            return result
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(perplexity, "evaluate", side_effect=fake_eval), \
             mock.patch.object(metrics_mod, "DEFAULT_METRICS_DIR", d), \
             mock.patch.dict(os.environ, {"SWI_METRICS": "1", "SWI_MODE": "TEST"}):
            code = perplexity.main(list(argv))
            path = os.path.join(d, "eval_ppl_TEST.jsonl")
            records = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
        return code, records

    RESULT = {"ckpt": "pretrain_checkpoint_TEST.pt", "ckpt_mtime": 1.0, "seq_len": 512,
              "heldout": {"tokens": 2000, "nll": 4.6, "ppl": 100.0}, "corpus_val": None}

    def test_gate_passes_and_logs_growth_record(self):
        code, records = self._run(self.RESULT, argv=["--max-ppl", "500"])
        self.assertEqual(code, 0)
        self.assertEqual(records[-1]["kind"], "eval")
        self.assertEqual(records[-1]["heldout_ppl"], 100.0)
        self.assertIsNone(records[-1]["corpus_val_ppl"])

    def test_gate_fails_above_threshold(self):
        code, _ = self._run(self.RESULT, argv=["--max-ppl", "50"])
        self.assertEqual(code, 1)

    def test_evaluation_crash_is_not_green(self):
        code, records = self._run(error=RuntimeError("corrupt checkpoint"))
        self.assertEqual(code, 1)
        self.assertEqual(records, [])


class TestCorpusValNotFakedFromTrain(unittest.TestCase):
    def test_corpus_val_is_none_without_a_held_out_split(self):
        with mock.patch("swi_llm.eval.run_eval.load_model", return_value=UniformModel()), \
             mock.patch.object(perplexity, "heldout_tokens", return_value=np.arange(100) % V), \
             mock.patch("swi_llm.data.token_corpus.load_token_corpus",
                        return_value=(np.arange(10_000) % V, np.zeros(0, dtype=np.uint16), {})):
            result = perplexity.evaluate(ckpt_path="none.pt")
        self.assertIsNone(result["corpus_val"])
        self.assertAlmostEqual(result["heldout"]["ppl"], V, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
