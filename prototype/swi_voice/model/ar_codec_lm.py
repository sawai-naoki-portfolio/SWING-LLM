"""
swi_voice/model/ar_codec_lm.py

ニューラルコーデック言語モデル方式（VALL-E 系）[音声編3.1]。

TTS を「次コーデックトークン予測」という LLM そのものの問題として扱う。
swi_llm の TransformerBlock（MLA + Sparse MoE + QK-Norm）をそのまま流用し、
語彙をテキストからコーデックトークンへ差し替えるだけでアーキテクチャ的には
通常の LLM と同一になる（LLM の技術基盤の転用）。

シーケンス = [テキスト埋め込み] + [フレーム埋め込み]。因果マスク下で、各フレーム
位置が「次フレームの各コードブックのトークン」を予測する。

ゼロショット音声クローニング [音声編4.1]:
  参照音声（数秒）のコーデックコード列を「音響プロンプト」としてフレーム系列の
  先頭に連結する。学習時はクリップ前半を音響プロンプト、後半をターゲットとし、
  ターゲット区間のみに損失をかける（prompt_frames でマスク）。推論時は参照音声の
  コードをプロンプトとして与えれば、追加学習なしにその声色でテキストを発話する。

生成コア:
  - サンプリング（temperature / top-k）: 貪欲 argmax の単調さ・繰り返しを回避。
  - KV キャッシュ: 各フレームで全系列を再計算する O(N^2) を回避し O(N) で自己回帰。

アーキテクチャ（埋め込み・ヘッド・BOS）は従来と不変のため既存チェックポイント互換。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_llm.core.models.transformer_blocks import (
    TransformerBlock, RMSNorm, precompute_freqs_cis, StaticMLACache,
)

logger = setup_logger("voice.ar_lm")


class ARCodecLM(nn.Module):
    def __init__(self, text_vocab_size, n_codebooks, codebook_size,
                 dim=512, n_layers=6, n_heads=8, hidden_dim=None,
                 num_experts=4, top_k=2, max_seq_len=2048, use_style=False):
        super().__init__()
        # スタイル/感情制御 [音声編4]。既定 False（C-1: 既存チェックポイント互換のため
        # style_proj を追加しない）。True のとき style_emb をテキスト条件へ加算する。
        self.use_style = use_style
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.bos_id = codebook_size                 # 入力フレーム系列の開始トークン（埋め込み用）
        # 停止制御 [音声編3.4]: 各ヘッドの出力語彙を codebook_size+1 に拡張し、末尾の
        # 追加クラス(index=codebook_size)を EOS とする。学習で実末尾に EOS を予測させ、
        # 生成では EOS 検出時に停止する（max_frames は安全上限に降格）。
        # EOS は入力には出さない（予測されたら止める）ため code_emb は拡張不要。
        self.eos_class = codebook_size              # ヘッド出力空間での EOS インデックス
        self.max_seq_len = max_seq_len

        self.text_emb = nn.Embedding(text_vocab_size, dim)
        # コードブックごとの埋め込み（+1 は BOS フレーム用。EOS は入力に現れないため不要）
        self.code_emb = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, dim) for _ in range(n_codebooks)])
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, hidden_dim, num_experts, top_k) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        # 出力は codebook_size + 1（実コード + EOS クラス）
        self.heads = nn.ModuleList(
            [nn.Linear(dim, codebook_size + 1) for _ in range(n_codebooks)])

        if use_style:
            self.style_proj = nn.Linear(dim, dim)

        head_dim = dim // n_heads
        fc, fs = precompute_freqs_cis(head_dim, max_seq_len * 2)
        # RoPE の cos/sin は buffer（state_dict 非対象）で ckpt 互換に影響しない
        self.register_buffer("freqs_cos", fc, persistent=False)
        self.register_buffer("freqs_sin", fs, persistent=False)

        logger.info(f"Initializing ARCodecLM: dim={dim}, layers={n_layers}, heads={n_heads}, "
                    f"n_codebooks={n_codebooks}, codebook={codebook_size}")

    def _text_embed(self, text_ids, style_emb=None):
        """テキスト埋め込み [B, Lt, dim]。use_style かつ style_emb 指定時はスタイルを全位置に加算。"""
        text_x = self.text_emb(text_ids)
        if self.use_style and style_emb is not None:
            text_x = text_x + self.style_proj(style_emb).unsqueeze(1)
        return text_x

    def _frame_embed(self, frame_codes):
        # frame_codes: [B, T, n_codebooks] -> [B, T, dim]（各段の埋め込みの和）
        return sum(self.code_emb[i](frame_codes[..., i]) for i in range(self.n_codebooks))

    def _run_layers(self, x, attn_mask=None):
        L = x.shape[1]
        fc, fs = self.freqs_cos[:L], self.freqs_sin[:L]
        aux_total = 0.0
        # attn_mask を渡すときは因果性もマスク側で表現するため is_causal=False にする。
        is_causal = attn_mask is None
        for layer in self.layers:
            x, aux, _ = layer(x, fc, fs, is_causal=is_causal, attn_mask=attn_mask)
            aux_total = aux_total + aux
        return self.norm(x), aux_total

    def _build_attn_mask(self, B, Lt, Ltot, text_mask, device):
        """因果マスク ∧「テキストの右パディング列を全クエリから無視」の 2D マスクを構築 (04#1)。

        text_mask [B, Lt]（True=実トークン）。フレーム部（先頭 Lt より後）は常に有効。
        query i は key j を「j<=i（因果）かつ key j が有効」のときのみ参照する。
        戻り値: [B, 1, Ltot, Ltot] の bool マスク（True=参照可）。
        """
        causal = torch.tril(torch.ones((Ltot, Ltot), device=device, dtype=torch.bool))
        key_valid = torch.ones((B, Ltot), dtype=torch.bool, device=device)
        key_valid[:, :Lt] = text_mask.to(torch.bool)
        return (causal.unsqueeze(0) & key_valid.unsqueeze(1)).unsqueeze(1)

    def forward(self, text_ids, codes, code_lens=None, prompt_frames=0, text_mask=None, style_emb=None):
        """text_ids: [B, Lt], codes: [B, Tf, n_codebooks]。教師強制で次フレームを予測。

        停止制御 [音声編3.4]: 入力を [BOS, codes]、ターゲットを [codes, EOS] とし、各サンプルの
        実末尾(code_lens)で EOS を予測させる。code_lens 指定時は右パディング分（実末尾より後）を
        損失から除外し、テキスト適応的な継続時間（可変長）を学習する。

        prompt_frames > 0 のとき、先頭 prompt_frames フレームを音響プロンプト（参照音声）と
        みなし損失外にする（クローニング: 声色を保った続きの生成のみを学習）。

        text_mask [B, Lt]（True=実トークン, False=右パディング）を渡すと、フレームが
        テキストのパディングトークンへ注意しないよう因果＋パディングマスクを適用する (04#1)。
        None の場合は従来どおり単純な因果マスク（後方互換）。
        """
        B, Lt = text_ids.shape
        T = codes.shape[1]
        device = codes.device
        if code_lens is None:
            code_lens = torch.full((B,), T, dtype=torch.long, device=device)

        text_x = self._text_embed(text_ids, style_emb)            # [B, Lt, dim]（任意でスタイル条件）
        bos = torch.full((B, 1, self.n_codebooks), self.bos_id, dtype=torch.long, device=device)
        in_codes = torch.cat([bos, codes], dim=1)                 # [B, T+1, n_q]（BOS + 全 codes）
        frame_x = self._frame_embed(in_codes)                     # [B, T+1, dim]
        x = torch.cat([text_x, frame_x], dim=1)                   # [B, Lt+T+1, dim]

        attn_mask = None
        if text_mask is not None:
            attn_mask = self._build_attn_mask(B, Lt, x.shape[1], text_mask, device)
        h, aux_total = self._run_layers(x, attn_mask=attn_mask)
        frame_h = h[:, Lt:, :]                                    # [B, T+1, dim] 各位置の予測
        logits = [head(frame_h) for head in self.heads]          # each [B, T+1, codebook_size+1]

        # ターゲット位置マスク: pos==len で EOS、pos<prompt_frames または pos>len は無視(-100)
        pos = torch.arange(T + 1, device=device).unsqueeze(0)     # [1, T+1]
        lens = code_lens.unsqueeze(1)                             # [B, 1]
        eos_mask = pos == lens                                    # [B, T+1]
        ignore_mask = (pos > lens) | (pos < prompt_frames)        # [B, T+1]
        codes_ext = torch.cat(                                    # [B, T+1, n_q]（末尾に placeholder）
            [codes, torch.zeros((B, 1, self.n_codebooks), dtype=torch.long, device=device)], dim=1)

        loss = 0.0
        C = self.codebook_size + 1
        for i in range(self.n_codebooks):
            tgt = codes_ext[..., i].clone()                      # [B, T+1]
            tgt[eos_mask] = self.eos_class                        # 実末尾で EOS を教師
            tgt[ignore_mask] = -100                               # プロンプト/パディングは損失外
            loss = loss + F.cross_entropy(
                logits[i].reshape(-1, C), tgt.reshape(-1), ignore_index=-100)
        loss = loss / self.n_codebooks + 0.01 * aux_total
        return logits, loss

    def _run_layers_cached(self, x, caches, start_pos, is_causal):
        """KV キャッシュを用いた層適用。start_pos は既にキャッシュ済みのフレーム数。

        MLA は xq に freqs[-L:]、xk に freqs[:full_seqlen] を使うため、freqs を
        [0 : start_pos + L] にスライスして渡す（キャッシュの seqlen==start_pos を前提）。
        初回（プロンプト一括処理）は is_causal=True、以降の 1 フレーム生成は
        is_causal=False（新規クエリは過去全キーを参照してよい）。
        """
        L = x.shape[1]
        fc = self.freqs_cos[: start_pos + L]
        fs = self.freqs_sin[: start_pos + L]
        for i, layer in enumerate(self.layers):
            x, _, _ = layer(x, fc, fs, is_causal=is_causal, use_cache=True, past_key_value=caches[i])
        return self.norm(x)

    def _sample_logits(self, logits, temperature, top_k):
        """logits [B, C] → サンプル済みインデックス [B]。temperature<=0 で貪欲。"""
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k and 0 < top_k < logits.shape[-1]:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return logits.argmax(dim=-1)

    def _sample_frame(self, last_h, temperature, top_k):
        """最終隠れ状態 [B, dim] から 1 フレームをサンプル。

        Returns: (codes[B, n_q], is_eos[B])。停止シグナルは codebook0 が EOS クラスか否か。
        codebook0 が EOS でなければその実コードを採用し、以降のブックは EOS クラスを除外して
        実コードのみをサンプルする（EOS を入力へ流さない）。
        """
        head_logits = [head(last_h) for head in self.heads]        # each [B, codebook_size+1]
        cb0 = self._sample_logits(head_logits[0], temperature, top_k)   # EOS 含む全クラスから
        is_eos = cb0 == self.eos_class                             # [B]
        frame = [cb0.clamp(max=self.codebook_size - 1)]            # EOS 時は停止するので値は不問
        for i in range(1, self.n_codebooks):
            real_logits = head_logits[i][..., : self.codebook_size]   # EOS クラス除外
            frame.append(self._sample_logits(real_logits, temperature, top_k))
        return torch.stack(frame, dim=-1), is_eos                  # [B, n_q], [B]

    @torch.no_grad()
    def generate(self, text_ids, prompt_codes=None, max_frames=200,
                 temperature=0.8, top_k=50, use_cache=True, style_emb=None):
        """テキスト（+参照音声プロンプト）からコード列 [B, T', n_codebooks] を自己回帰生成。

        停止制御 [音声編3.4]: codebook0 が EOS クラスを出した時点で停止する。max_frames は
        EOS が出ないときの安全上限であり、通常はテキスト内容に応じて早期に停止する。

        Args:
            prompt_codes: [B, Lp, n_q]。参照音声のコーデックコード列（ゼロショット
                クローニングの音響プロンプト）。None なら通常の text→codes 生成。
            max_frames: 生成フレーム数の安全上限（EOS が出れば手前で停止）。
            temperature/top_k: サンプリング設定（temperature<=0 で貪欲）。
            use_cache: KV キャッシュで O(N) 生成（False なら毎回全再計算の O(N^2)）。

        Returns:
            新規生成したフレームのみ [B, T', n_q]（プロンプト/EOS は含めない）。
        """
        self.eval()
        B = text_ids.shape[0]
        device = text_ids.device

        bos = torch.full((B, 1, self.n_codebooks), self.bos_id, dtype=torch.long, device=device)
        if prompt_codes is not None and prompt_codes.shape[1] > 0:
            init_frames = torch.cat([bos, prompt_codes.to(device)], dim=1)   # [B, 1+Lp, n_q]
        else:
            init_frames = bos                                                # [B, 1, n_q]

        Lt = text_ids.shape[1]
        init_len = init_frames.shape[1]
        text_x = self._text_embed(text_ids, style_emb)

        if not use_cache:
            # フォールバック: キャッシュ無しの O(N^2) 生成（参照実装 / 検証用）
            frames = init_frames
            generated = []
            for _ in range(max_frames):
                x = torch.cat([text_x, self._frame_embed(frames)], dim=1)
                if x.shape[1] >= self.max_seq_len:
                    break
                h, _ = self._run_layers(x)
                next_codes, is_eos = self._sample_frame(h[:, -1, :], temperature, top_k)
                if bool(is_eos.all()):                            # EOS で停止（max_frames は上限）
                    break
                generated.append(next_codes)
                frames = torch.cat([frames, next_codes.unsqueeze(1)], dim=1)
            if not generated:
                return torch.zeros((B, 0, self.n_codebooks), dtype=torch.long, device=device)
            return torch.stack(generated, dim=1)

        # KV キャッシュ経路: 内部ジェネレータでフレームを逐次生成して結合
        frames = list(self._generate_frames_cached(
            text_ids, prompt_codes, max_frames, temperature, top_k, style_emb=style_emb))
        if not frames:
            return torch.zeros((B, 0, self.n_codebooks), dtype=torch.long, device=device)
        return torch.stack(frames, dim=1)                                    # [B, T', n_q]

    def _generate_frames_cached(self, text_ids, prompt_codes, max_frames, temperature, top_k, style_emb=None):
        """KV キャッシュで 1 フレームずつ yield する内部ジェネレータ（EOS で停止）。

        generate（一括）と generate_stream（チャンク）で共有し、生成フレーム列の一貫性を保つ。
        no_grad は generator でも yield 跨ぎで保持されるよう with で囲う（デコレータは不可）。
        """
        with torch.no_grad():
            self.eval()
            B = text_ids.shape[0]
            device = text_ids.device
            bos = torch.full((B, 1, self.n_codebooks), self.bos_id, dtype=torch.long, device=device)
            if prompt_codes is not None and prompt_codes.shape[1] > 0:
                init_frames = torch.cat([bos, prompt_codes.to(device)], dim=1)
            else:
                init_frames = bos
            Lt = text_ids.shape[1]
            init_len = init_frames.shape[1]
            text_x = self._text_embed(text_ids, style_emb)

            latent_dim = getattr(self.layers[0].attention, "latent_dim", 512)
            dtype = self.text_emb.weight.dtype
            cache_cap = min(self.max_seq_len, Lt + init_len + max_frames + 4)
            caches = [StaticMLACache(B, cache_cap, latent_dim, device, dtype) for _ in self.layers]

            # 初回: [text + BOS + プロンプト] を一括で処理（因果マスク）
            x0 = torch.cat([text_x, self._frame_embed(init_frames)], dim=1)
            h = self._run_layers_cached(x0, caches, start_pos=0, is_causal=True)
            last = h[:, -1, :]

            n_gen = 0
            for _ in range(max_frames):
                next_codes, is_eos = self._sample_frame(last, temperature, top_k)   # [B, n_q], [B]
                if bool(is_eos.all()):                                       # EOS で停止
                    return
                yield next_codes                                            # [B, n_q]
                n_gen += 1
                start_pos = Lt + init_len + n_gen - 1                       # 現在キャッシュ長
                if start_pos >= cache_cap - 1:
                    return
                x = self._frame_embed(next_codes.unsqueeze(1))              # [B, 1, dim]
                h = self._run_layers_cached(x, caches, start_pos=start_pos, is_causal=False)
                last = h[:, -1, :]

    def generate_stream(self, text_ids, prompt_codes=None, chunk_frames=16,
                        max_frames=1000, temperature=0.8, top_k=50, style_emb=None):
        """フレームを chunk_frames ごとにまとめて逐次 yield する（ストリーミング推論 [音声編7.1]）。

        各 yield は [B, c, n_q]（c<=chunk_frames）。EOS で停止。先頭チャンクを即座に返せるため
        後段（コーデックデコード→再生）と並行でき、TTFB（初回音声までの時間）を短縮する。

        style_emb は generate と同じくテキスト条件へ加算する（感情タグ・参照音声・自然言語の
        スタイル指示 [音声編6.3]をストリーミング経路でも反映させる）。
        """
        buf = []
        for frame in self._generate_frames_cached(
                text_ids, prompt_codes, max_frames, temperature, top_k, style_emb=style_emb):
            buf.append(frame)
            if len(buf) >= chunk_frames:
                yield torch.stack(buf, dim=1)                               # [B, chunk_frames, n_q]
                buf = []
        if buf:
            yield torch.stack(buf, dim=1)
