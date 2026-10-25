import os
import time

import torch

from swi_llm.utils.logger import setup_logger

# setup_logger 経由にすることで app.log/error.log へ確実に出力し、SWI_LOG_LEVEL=DEBUG での
# 深掘りを可能にする（従来の logging.getLogger はハンドラ未設定で app.log に届かなかった）。
logger = setup_logger("trainer_utils")

def calculate_save_steps(total_steps, save_interval_ratio, min_steps=1):
    """
    Calculate the number of steps before saving an intra-epoch checkpoint.
    """
    save_steps = int(total_steps * save_interval_ratio)
    return max(min_steps, save_steps)


def maybe_wrap_fsdp(model):
    """問題5（大規模化）: torch.distributed が初期化済み（torchrun 等）で world_size>1 の
    ときだけ FSDP (FULL_SHARD) でモデルを分割する。単一プロセス/単一デバイス（通常運用・テスト）
    では wrap せず素通しし、従来挙動を完全に維持する（分散未初期化なら何もしない）。

    注: FSDP 有効時のチェックポイント保存は sharded state_dict の集約 API が別途必要で、
    現状は単一デバイス保存を前提とする（分散保存対応は今後の課題）。"""
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return model
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        logger.info(f"Wrapping model with FSDP (world_size={dist.get_world_size()}, FULL_SHARD).")
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
    except Exception as e:
        logger.warning(f"FSDP wrap skipped ({e}); using single-device model.")
        return model


def maybe_compile(model):
    """問題5（効率化）: SWI_USE_TORCH_COMPILE=1 のとき torch.compile でカーネル融合を有効化する
    （既定 OFF）。Windows/Triton 非対応環境や未対応 PyTorch では失敗しうるため try/except で
    eager モデルへフォールバックする。既定では素通し（従来挙動を維持）。"""
    if os.environ.get("SWI_USE_TORCH_COMPILE", "").strip().lower() not in ("1", "true", "yes"):
        return model
    try:
        compiled = torch.compile(model)
        logger.info("Applied torch.compile (kernel fusion enabled).")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile skipped ({e}); using eager model.")
        return model


def _strip_compiled_prefix(state_dict):
    """torch.compile 済みモデルの state_dict は '_orig_mod.' 接頭辞が付くため、保存前に除去して
    非 compile モデルと互換なキーへ正規化する（未 compile 時は no-op）。"""
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict):
        return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                for k, v in state_dict.items()}
    return state_dict

class CheckpointLoadError(RuntimeError):
    """存在するチェックポイントを読めなかった（破損・構造不一致）。

    L-17: 以前はここで例外を握り潰して「スクラッチ開始」扱いにしていたため、数週間の学習の
    ckpt が壊れると黙って最初からやり直し、次の保存で既存 ckpt を上書きしていた。
    存在するのに読めない ckpt は必ず止めて人に判断させる（--fresh で破棄 or 復旧）。
    """


def ckpt_save_interval_seconds():
    """時間ベースの定期保存間隔（秒）。0 以下で無効。SWI_CKPT_SAVE_MINUTES で上書き。"""
    from swi_llm.utils.constants import CKPT_SAVE_INTERVAL_MINUTES
    raw = os.environ.get("SWI_CKPT_SAVE_MINUTES", "").strip()
    try:
        minutes = float(raw) if raw else float(CKPT_SAVE_INTERVAL_MINUTES)
    except ValueError:
        logger.warning(f"Invalid SWI_CKPT_SAVE_MINUTES={raw!r}; using {CKPT_SAVE_INTERVAL_MINUTES}.")
        minutes = float(CKPT_SAVE_INTERVAL_MINUTES)
    return minutes * 60.0


def _load_state_dict_checked(model, state_dict, ckpt_path):
    """strict=False でロードしつつ、1 つも一致しない（= 実質ランダム重みのまま）なら止める。

    torch.compile 済みモデル（SWI_USE_TORCH_COMPILE=1）は state_dict のキーに '_orig_mod.' が付き、
    保存時に剥がした ckpt と 1 つも一致しない（以前は再開時に黙ってランダム重みのままだった）。
    中身のモジュールへロードする。
    """
    target = getattr(model, "_orig_mod", model)
    result = target.load_state_dict(state_dict, strict=False)
    if state_dict and len(result.unexpected_keys) == len(state_dict):
        raise CheckpointLoadError(
            f"None of the {len(state_dict)} tensors in {ckpt_path} match the model "
            f"(architecture changed?). Refusing to continue from random weights.")
    return result


def load_checkpoint(ckpt_path, device, model, optimizer=None, scheduler=None, scaler=None):
    """
    Loads a checkpoint and restores the model and training state.

    Raises:
        CheckpointLoadError: ckpt_path は存在するが読めない（破損・構造不一致）。
            ファイルが無い場合だけがスクラッチ開始（L-17）。
    Returns:
        start_epoch (int): Epoch to resume from.
        start_step (int): Step to resume from.
        best_val_loss (float): Best validation loss recorded.
        resume_checkpoint (dict): The loaded raw checkpoint if needed.
    """
    if not os.path.exists(ckpt_path):
        logger.info(f"Checkpoint {ckpt_path} not found. Starting from scratch.")
        return 0, 0, float('inf'), None

    logger.info(f"Loading checkpoint from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Determine if it's a raw state dict or a full training checkpoint
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            logger.debug(f"Checkpoint format: full training ckpt (has model_state_dict) at {ckpt_path}")
            state_dict = {k: v for k, v in ckpt['model_state_dict'].items() if not k.endswith("freqs_cos") and not k.endswith("freqs_sin")}
            _load_state_dict_checked(model, state_dict, ckpt_path)
        elif isinstance(ckpt, dict) and 'vision_encoder' in ckpt:
            # Special case for Vision training
            logger.debug(f"Checkpoint format: vision ckpt (vision_encoder/projector) at {ckpt_path}")
            model.vision_encoder.load_state_dict(ckpt['vision_encoder'])
            model.vision_projector.load_state_dict(ckpt['vision_projector'])
        else:
            logger.debug(f"Checkpoint format: raw state_dict (base weights, no training state) at {ckpt_path}")
            state_dict = {k: v for k, v in ckpt.items() if not k.endswith("freqs_cos") and not k.endswith("freqs_sin")}
            _load_state_dict_checked(model, state_dict, ckpt_path)
            # 生の state_dict（ベース重み）を実際にロード済み。ckpt を返して呼び出し側が
            # 「未ロード」と誤認し、フォールバック/pretrain 重みで上書きしてしまうのを防ぐ
            # （学習状態は無いので epoch/step は 0 から）。
            return 0, 0, float('inf'), ckpt
            
        logger.info("Model weights loaded successfully.")
        
        start_epoch = ckpt.get('epoch', 0)
        start_step = ckpt.get('step', -1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        # Restore training states
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        elif optimizer and 'optimizer' in ckpt:
             optimizer.load_state_dict(ckpt['optimizer'])
             
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        elif scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        elif scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            
        if optimizer:
            logger.info(f"Resuming from epoch {start_epoch}, step {start_step} with best_val_loss {best_val_loss:.4f}")
            
        torch.cuda.empty_cache()
        return start_epoch, start_step, best_val_loss, ckpt
        
    except CheckpointLoadError:
        raise
    except Exception as e:
        raise CheckpointLoadError(
            f"Failed to load existing checkpoint {ckpt_path}: {e}. NOT starting from scratch "
            f"(that would overwrite it on the next save). Restore the file, or discard it "
            f"explicitly with `run.bat ... --fresh`.") from e

def save_checkpoint(ckpt_path, model_state, optimizer, scheduler, scaler, epoch, step, best_val_loss,
                    data_size=None):
    """
    Saves a checkpoint containing the model and training state.
    model_state can be model.state_dict() or a custom dictionary (e.g. vision).

    data_size: 学習に使ったデータ件数。`BaseTrainer.is_training_complete()` が
        「データが変わったのに完了扱いで学習がスキップされる」静かな失敗を検知するために保存する
        （None なら保存しない = 旧チェックポイント互換）。
    """
    try:
        ckpt_state = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }
        if data_size is not None:
            ckpt_state['data_size'] = data_size
        
        # Merge model state
        if isinstance(model_state, dict) and ('vision_encoder' in model_state or 'model_state_dict' in model_state):
             ckpt_state.update(model_state)
        else:
             # 問題5: torch.compile 済みモデルの '_orig_mod.' 接頭辞を除去して ckpt 互換を保つ。
             ckpt_state['model_state_dict'] = _strip_compiled_prefix(model_state)

        # L-17: 一時ファイルへ書き切ってから置き換える（アトミック）。保存中の停電・強制終了でも
        # 既存の ckpt は壊れず、前回の保存点から再開できる。
        tmp_path = f"{ckpt_path}.tmp"
        torch.save(ckpt_state, tmp_path)
        os.replace(tmp_path, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
    except Exception as e:
        # L-17: 保存失敗（ディスク満杯等）を握り潰すと、数日分の学習が保存されないまま進む。
        # 止めて知らせる（直前の ckpt はアトミック保存により無傷で、そこから再開できる）。
        logger.error(f"Failed to save checkpoint at {ckpt_path}: {e}")
        try:
            os.remove(f"{ckpt_path}.tmp")
        except OSError:
            pass
        raise

class BaseTrainer:
    """
    A unified base class to encapsulate boilerplate training setup logic
    and reduce duplication across training scripts.
    """
    def __init__(self, cfg_name, main_ckpt_path, fallback_ckpt_paths=None, model_fn=None):
        self.logger = setup_logger(cfg_name + "_trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        from swi_llm.utils.config import config, TRAINING_CONFIG, create_mini_llm
        from swi_llm.data.tokenizer import VOCAB_SIZE
        import torch.optim as optim
        
        self.cfg_name = cfg_name
        self.global_config = config
        self.cfg = TRAINING_CONFIG[cfg_name]
        
        self.seq_len = self.cfg["seq_len"]
        self.batch_size = self.cfg["batch_size"]
        self.grad_accum_steps = self.cfg["grad_accum_steps"]
        self.epochs = self.cfg["epochs"]
        self.lr = self.cfg["lr"]
        
        # Model Init
        if model_fn is not None:
            self.model = model_fn().to(self.device)
        else:
            self.model = create_mini_llm(vocab_size=VOCAB_SIZE, max_seq_len=self.seq_len).to(self.device)

        # 問題5: 分散環境では FSDP でシャーディング（optimizer 生成前に wrap する必要がある）。
        # 単一デバイス/既定では素通しで従来挙動を維持。
        self.model = maybe_wrap_fsdp(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # 問題5: SWI_USE_TORCH_COMPILE=1 で torch.compile（既定 OFF）。params は共有のため
        # optimizer 生成後でも可。保存時は _orig_mod. 接頭辞を除去して ckpt 互換を保つ。
        self.model = maybe_compile(self.model)
        
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and self.amp_dtype==torch.float16))

        self.main_ckpt_path = main_ckpt_path
        self.fallback_ckpt_paths = fallback_ckpt_paths or []

        self.start_epoch = 0
        self.start_step = 0
        self.best_val_loss = float('inf')
        self.scheduler = None
        # データ変更検知用（is_training_complete(data_size=...) で設定・比較する）。
        self.data_size = None            # 今回の学習データ件数
        self.loaded_data_size = None     # ロードした ckpt が学習した件数（旧 ckpt では None）
        # L-17: 時間ベースの定期保存。比率ベースだけだと数週間の 1 エポックで保存が数日おきになる。
        self.save_interval_seconds = ckpt_save_interval_seconds()
        self._last_save_time = time.monotonic()

        # NEW-Q2 (S-3): 学習メトリクスの時系列（loss/lr/GPU メモリ/スループット）を
        # errorLog/metrics/ へ JSONL で常設記録する。step_and_save が optimizer ステップ
        # ごとに記録するので、BaseTrainer を使う全 trainer が自動的に対象になる。
        from swi_llm.utils.metrics import MetricsLogger
        self.metrics = MetricsLogger(cfg_name)

        # NEW-L4: SDPA の flash カーネルを明示有効化し、実効状態を起動ログに残す
        # （初回のみログ。Windows では Triton 版 FlashAttention の代替として SDPA 内蔵を使う）。
        from swi_llm.utils.attention_backends import configure_sdp_backends
        configure_sdp_backends()

        # DEBUG: 学習セットアップの解決結果（SWI_LOG_LEVEL=DEBUG で再現条件を把握するため）
        self.logger.debug(
            f"Trainer[{cfg_name}] setup: seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"grad_accum={self.grad_accum_steps}, epochs={self.epochs}, lr={self.lr}, "
            f"amp_dtype={self.amp_dtype}, grad_scaler_enabled={self.scaler.is_enabled()}, "
            f"model_fn={'custom' if model_fn else 'create_mini_llm'}, "
            f"fallbacks={len(self.fallback_ckpt_paths)}")

    def setup_scheduler(self, train_loader_len):
        from swi_llm.utils.optim import get_cosine_schedule_with_warmup
        total_steps = (train_loader_len // self.grad_accum_steps) * self.epochs
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        # DEBUG: スケジューラの決定パラメータ（総ステップ・ウォームアップ）
        self.logger.debug(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps} "
                          f"(loader_len={train_loader_len}, grad_accum={self.grad_accum_steps}, epochs={self.epochs})")

    def load_checkpoints(self, disable_pretrain_fallback=False):
        # Try main checkpoint
        start_epoch, start_step, best_val_loss, ckpt = load_checkpoint(
            self.main_ckpt_path, self.device, self.model, self.optimizer, self.scheduler, self.scaler
        )
        
        if ckpt is None and self.fallback_ckpt_paths:
            # Try fallbacks
            for path in self.fallback_ckpt_paths:
                self.logger.info(f"Attempting to load fallback checkpoint: {path}")
                is_same_stage = self.cfg_name in os.path.basename(path)
                
                start_epoch, start_step, best_val_loss, ckpt = load_checkpoint(
                    path, self.device, self.model, 
                    self.optimizer if is_same_stage else None, 
                    self.scheduler if is_same_stage else None, 
                    self.scaler if is_same_stage else None
                )
                if ckpt is not None:
                    if not is_same_stage:
                        self.logger.info(f"Fallback checkpoint is from a previous stage. Resetting epoch and step.")
                        start_epoch = 0
                        start_step = 0
                        best_val_loss = float('inf')
                    break
                    
        # If still no full ckpt, try loading base weights (pretrain)
        if ckpt is None and not disable_pretrain_fallback:
            from swi_llm.utils.constants import CKPT_PRETRAIN, EXT_PT
            pretrain_path = f"{CKPT_PRETRAIN}{self.global_config['ckpt_suffix']}{EXT_PT}"
            if os.path.exists(pretrain_path) and self.main_ckpt_path != pretrain_path:
                self.logger.info(f"Starting from pretrain checkpoint {pretrain_path}...")
                load_checkpoint(pretrain_path, self.device, self.model)
            else:
                self.logger.warning("No valid training checkpoints found. Starting from scratch or base.")
            
            start_epoch, start_step, best_val_loss = 0, 0, float('inf')
        elif ckpt is None:
            self.logger.warning("No valid training checkpoints found. Starting from scratch or custom base.")
            start_epoch, start_step, best_val_loss = 0, 0, float('inf')
            
        self.start_epoch = start_epoch
        self.start_step = start_step
        self.best_val_loss = best_val_loss
        # 完了扱いによる「学習データ変更の握り潰し」検知に使う（is_training_complete 参照）。
        self.loaded_data_size = ckpt.get('data_size') if isinstance(ckpt, dict) else None
        # DEBUG: 再開点の解決結果（どこから学習を続けるか）
        self.logger.debug(f"Resolved training start: epoch={start_epoch}, step={start_step}, "
                          f"best_val_loss={best_val_loss}, full_ckpt_loaded={ckpt is not None}, "
                          f"loaded_data_size={self.loaded_data_size}")
        return ckpt

    def get_save_interval(self, train_loader_len):
        interval = calculate_save_steps(train_loader_len // self.grad_accum_steps, self.cfg.get("save_interval_ratio", 0.25))
        self.logger.debug(f"Save interval: every {interval} optimizer-steps "
                          f"(opt_steps_per_epoch={train_loader_len // self.grad_accum_steps})")
        return interval

    def step_and_save(self, loss, step, dataloader_len, epoch, save_interval_steps, save_path=None, skip_backward=False):
        if not skip_backward:
            self.scaler.scale(loss).backward()
        
        if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == dataloader_len:
            self.scaler.unscale_(self.optimizer)
            import torch
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            opt_step = (step + 1) // self.grad_accum_steps
            # NEW-Q2 (S-3): optimizer ステップ単位のメトリクス記録（JSONL・コンソール非出力）。
            self.metrics.log_step(
                epoch=epoch, step=opt_step,
                loss=float(loss.detach()) if loss is not None else None,
                lr=self.scheduler.get_last_lr()[0],
                n_samples=self.batch_size * self.grad_accum_steps)
            due_by_steps = opt_step > 0 and opt_step % save_interval_steps == 0
            interval_s = getattr(self, "save_interval_seconds", 0)
            due_by_time = interval_s > 0 and (
                time.monotonic() - getattr(self, "_last_save_time", time.monotonic()) >= interval_s)
            if due_by_steps or due_by_time:
                path_to_save = save_path if save_path else self.main_ckpt_path
                save_checkpoint(path_to_save, self.model.state_dict(), self.optimizer, self.scheduler, self.scaler, epoch, step, self.best_val_loss, data_size=self.data_size)
                self._last_save_time = time.monotonic()

    def is_training_complete(self, data_size=None):
        """完了済みチェックポイントなら True（学習をスキップする）。

        Args:
            data_size: 今回の学習データ件数（例: `len(train_loader.dataset)`）。
                渡すと、ckpt が学習した件数と比較し、**データが変わっているのに
                「完了」でスキップされる**静かな失敗を WARNING で可視化する。
                （実際に遭遇した例: SFT へ tool_call 教師 180 件を追加したが、
                 `epoch 3 >= 3` で即終了し新データが 1 件も学習されなかった。
                 TEST/SMOKE は run.ps1 が既定で fresh のため露見しないが、
                 PROD/MAX は既定 resume なので黙って握り潰される。）
                データが変わっていても**自動では再学習しない**（PROD で数時間〜数日の
                再学習が意図せず走る方が危険なため）。`run.bat ... --fresh` を促す。
        """
        if data_size is not None:
            self.data_size = data_size

        if self.start_epoch >= self.epochs:
            if (data_size is not None and self.loaded_data_size is not None
                    and data_size != self.loaded_data_size):
                self.logger.warning(
                    f"Training data changed ({self.loaded_data_size} -> {data_size} samples) "
                    f"but the checkpoint is already complete (epoch {self.start_epoch} >= {self.epochs}); "
                    f"training will be SKIPPED and the new data will NOT be learned. "
                    f"Re-run with --fresh (or delete {os.path.basename(self.main_ckpt_path)}) "
                    f"to train on the updated dataset.")
            self.logger.info(f"Training already complete (epoch {self.start_epoch} >= {self.epochs}).")
            return True
        return False

