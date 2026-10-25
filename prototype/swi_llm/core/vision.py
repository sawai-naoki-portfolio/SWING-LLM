import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.core.models.transformer_blocks import RMSNorm, SwiGLU
from swi_llm.utils.logger import setup_logger

logger = setup_logger("core.vision")

class PatchEmbedding(nn.Module):
    """画像を固定サイズのパッチに分割し、線形変換を行うクラス"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.in_channels = in_channels
        
        # Conv2dの代わりにLinearとUnfoldを用いて同じ処理を行う（cudnnバグ回避のため）
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        
        # クラストークンと位置エンコーディング
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        
        # 画像をパッチに分割 (Unfold)
        # x: [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        x = x.unfold(2, p, p).unfold(3, p, p)
        
        # [B, C, num_patches, P, P]
        x = x.contiguous().view(bsz, self.in_channels, -1, p, p)
        
        # [B, num_patches, C, P, P]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # [B, num_patches, C * P * P]
        x = x.view(bsz, self.num_patches, -1)
        
        # [B, num_patches, embed_dim]
        x = self.proj(x)
        
        # クラストークンを追加
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 位置エンコーディングを加算
        x = x + self.pos_embed
        return x

class VisionAttention(nn.Module):
    """Vision Transformer用のMulti-Head Attention"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Performance optimization: Use scaled_dot_product_attention instead of manual matmul
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        x = self.proj(x)
        return x

class VisionEncoderBlock(nn.Module):
    """Vision Transformerの1ブロック"""
    def __init__(self, dim, n_heads, mlp_hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionEncoder(nn.Module):
    """Vision Transformer全体"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, n_heads=12, mlp_hidden_dim=3072):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            VisionEncoderBlock(embed_dim, n_heads, mlp_hidden_dim)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class VideoFrameEncoder(nn.Module):
    """動画（フレーム列）の時空間エンコード [02_multimodal 展望2, NEW-L5]。

    factorized 時空間アテンションの最小形（TimeSformer の divided attention と同型の発想）:
      1. **空間**: 各フレームを既存 VisionEncoder で独立にエンコード（重み共有）。
      2. フレームごとに空間トークンを平均プーリングして「フレームトークン」1 個へ圧縮
         （T フレームで T トークン = LLM コンテキストを時間方向に線形でしか消費しない）。
      3. **時間**: 学習可能な時間位置埋め込みを加え、時間方向の Transformer ブロック
         （VisionEncoderBlock を流用）でフレーム間を相互参照させる。

    出力 [B, T, embed_dim] は画像経路と同じ VisionProjector で LLM 次元へ射影でき、
    `SWIVisionLanguageModel(videos=...)` から vision トークン列として注入される。
    学習（video-caption データでの projector/temporal 学習）は将来の投資ステップ。
    """

    def __init__(self, vision_encoder, embed_dim=768, max_frames=32,
                 temporal_depth=2, n_heads=8):
        super().__init__()
        self.vision_encoder = vision_encoder                 # 画像経路と重み共有
        self.max_frames = max_frames
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        nn.init.normal_(self.temporal_pos, std=0.02)
        self.temporal_blocks = nn.ModuleList([
            VisionEncoderBlock(embed_dim, n_heads, embed_dim * 4)
            for _ in range(temporal_depth)
        ])
        self.norm = RMSNorm(embed_dim)
        logger.info(f"Initializing VideoFrameEncoder: max_frames={max_frames}, "
                    f"temporal_depth={temporal_depth}, embed_dim={embed_dim}")

    def forward(self, frames):
        """frames [B, T, C, H, W] → 時空間特徴 [B, T, embed_dim]。"""
        if frames.dim() != 5:
            raise ValueError(f"videos must be [B, T, C, H, W]; got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        if t > self.max_frames:
            logger.warning(f"VideoFrameEncoder: {t} frames > max_frames={self.max_frames}; "
                           "先頭から等間隔に間引きます。")
            idx = torch.linspace(0, t - 1, self.max_frames, device=frames.device).long()
            frames = frames[:, idx]
            t = self.max_frames
        feats = self.vision_encoder(frames.reshape(b * t, *frames.shape[2:]))   # [B*T, P+1, D]
        frame_tokens = feats.mean(dim=1).view(b, t, -1)                          # 空間平均プール
        x = frame_tokens + self.temporal_pos[:, :t]
        for block in self.temporal_blocks:                                       # 時間アテンション
            x = block(x)
        return self.norm(x)                                                      # [B, T, D]


class VisionProjector(nn.Module):
    """Vision Encoderの出力をLLMの入力次元（dim）にマッピングする2層MLP"""
    def __init__(self, vision_dim=768, llm_dim=512):
        super().__init__()
        # SwiGLU風やGELUなど様々な設計が可能だが、ここではシンプルなMLP (CLIP-like) + GELU
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim, bias=False),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False)
        )

    def forward(self, x):
        # x: [B, seq_len, vision_dim] -> [B, seq_len, llm_dim]
        return self.proj(x)

def preprocess_pil_image(img, target_size=None, policy=None):
    """PIL 画像 → 正規化済みテンソル [1, 3, H, W]。

    01(小): 学習(vision_data.py)・推論(web/models.parse_images)・本ファイルの
    process_image で重複していた前処理を単一情報源に統合する（L-6 の正規化統計
    統一と同趣旨）。リサイズ方針は VISION_CONFIG["resize_policy"] を参照:
      - "squash"        : アスペクト比を無視して直接リサイズ（現行学習分布。既定）
      - "shortside_crop": 短辺基準リサイズ(BICUBIC) + センタークロップ（CLIP 標準・歪みなし）
    切替は vision 再学習とセットで行うこと（roadmap/llm ステップ7）。
    """
    import numpy as np
    from PIL import Image
    from swi_llm.utils.config import VISION_CONFIG

    t_h, t_w = tuple(target_size or VISION_CONFIG["image_size"])
    policy = policy or VISION_CONFIG.get("resize_policy", "squash")

    if img.mode != "RGB":
        img = img.convert("RGB")

    if policy == "shortside_crop":
        w, h = img.size
        scale = max(t_w / w, t_h / h)               # 短辺がターゲットを満たす倍率
        new_w = max(t_w, round(w * scale))
        new_h = max(t_h, round(h * scale))
        resampling = getattr(Image, "Resampling", Image)
        img = img.resize((new_w, new_h), resampling.BICUBIC)
        left = (new_w - t_w) // 2
        top = (new_h - t_h) // 2
        img = img.crop((left, top, left + t_w, top + t_h))
    else:
        # 既定 squash: 旧実装と同一挙動（resample 引数も渡さない）を厳密に維持する
        img = img.resize((t_w, t_h))

    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))      # [H,W,C] -> [C,H,W]
    tensor = torch.tensor(img_arr).unsqueeze(0)

    # CLIP 正規化基準（VISION_CONFIG）— 学習・推論と同一の単一情報源（L-6）
    mean = torch.tensor(VISION_CONFIG["image_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(VISION_CONFIG["image_std"]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def process_image(image_path_or_tensor, target_size=(224, 224)):
    """
    画像データ（パス or テンソル）を受け取り、モデルに入力可能なテンソルへ変換する。
    実処理は preprocess_pil_image に委譲（リサイズ方針は VISION_CONFIG が単一情報源）。

    01(小): 旧実装は PIL 欠如時に print + torch.randn のダミーテンソルを返しており、
    無意味な入力が黙って下流へ流れる状態だった。fail-loud 方針（V-2/REQ 系）に合わせ、
    依存欠如は明示的な RuntimeError にする。
    """
    if isinstance(image_path_or_tensor, torch.Tensor):
        # 既にテンソルの場合（テスト・上流で前処理済みの経路）
        return image_path_or_tensor

    try:
        from PIL import Image
    except ImportError as e:
        logger.error("Pillow (PIL) がインストールされていないため画像を処理できません。")
        raise RuntimeError(
            "process_image requires Pillow. Install it (pip install Pillow) — "
            "returning dummy tensors on missing dependencies is disabled (fail-loud)."
        ) from e

    img = Image.open(image_path_or_tensor)
    return preprocess_pil_image(img, target_size=target_size)

def process_image_anyres(image_tensor, target_size=(224, 224)):
    """
    画像の縦横比を維持しながら、固定解像度(target_size)のグリッドに分割し、
    グローバルコンテキスト画像と合わせて複数のパッチ画像を生成する (AnyRes処理)
    
    Args:
        image_tensor: [B, C, H, W] の画像テンソル
        target_size: 分割先の基本解像度 (H_t, W_t)
        
    Returns:
        [B, num_grids + 1, C, H_t, W_t] のテンソル
    """
    bsz, c, h, w = image_tensor.shape
    t_h, t_w = target_size
    
    # 1. グローバルコンテキスト用（全体をターゲットサイズにリサイズしたもの）
    global_img = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    # 2. グリッド分割数の計算
    grid_h = max(1, h // t_h + (1 if h % t_h != 0 else 0))
    grid_w = max(1, w // t_w + (1 if w % t_w != 0 else 0))
    
    # 必要に応じてパディングしてグリッドサイズに合わせる
    pad_h = (grid_h * t_h) - h
    pad_w = (grid_w * t_w) - w
    if pad_h > 0 or pad_w > 0:
        image_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h))
    else:
        image_padded = image_tensor
        
    # パッチに分割 [B, C, grid_h * t_h, grid_w * t_w] -> [B, grid_h*grid_w, C, t_h, t_w]
    patches = image_padded.unfold(2, t_h, t_h).unfold(3, t_w, t_w)
    patches = patches.contiguous().view(bsz, c, grid_h * grid_w, t_h, t_w)
    patches = patches.permute(0, 2, 1, 3, 4)
    
    # グローバル画像とパッチ画像を結合: [B, 1 + grid_h*grid_w, C, t_h, t_w]
    global_img = global_img.unsqueeze(1)
    output = torch.cat([global_img, patches], dim=1)
    
    return output


if __name__ == '__main__':
    # ==========================================
    # テストと検証 (ダミーテンソルによるShape確認)
    # ==========================================
    print("--- Vision Module Integration Test ---")
    
    # 1. パラメータの設定
    batch_size = 2
    image_size = 224
    patch_size = 16
    in_channels = 3
    vision_dim = 768
    llm_dim = 512 # prototype/model.py のデフォルト MiniLLM dim に合わせる
    
    print(f"Configuration: Image={image_size}x{image_size}, Patch={patch_size}, VisionDim={vision_dim}, LLMDim={llm_dim}")
    
    # 2. モデルのインスタンス化
    vision_encoder = VisionEncoder(
        image_size=image_size, 
        patch_size=patch_size, 
        in_channels=in_channels, 
        embed_dim=vision_dim, 
        depth=4,       # テスト用に軽量化
        n_heads=8, 
        mlp_hidden_dim=vision_dim * 4
    )
    vision_projector = VisionProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    
    # 3. ダミー画像（高解像度・非正方形）の生成とAnyRes前処理
    # 例として 800 x 400 の長方形画像を入力とする
    dummy_image_highres = torch.randn(batch_size, in_channels, 800, 400)
    print(f"Original high-res image shape: {dummy_image_highres.shape}")
    
    # AnyRes処理: 224x224のグリッドに分割。
    # 800 // 224 + 1 = 4 (高さグリッド), 400 // 224 + 1 = 2 (幅グリッド) -> 計 8パッチ + 1(グローバル) = 9
    processed_images_anyres = process_image_anyres(dummy_image_highres, target_size=(image_size, image_size))
    print(f"AnyRes processed images shape: {processed_images_anyres.shape}")
    
    # バッチ内の各パッチをエンコーダに流し込むためにreshape: [B, num_grids, C, H, W] -> [B * num_grids, C, H, W]
    b, n_grids, c, h, w = processed_images_anyres.shape
    anyres_input = processed_images_anyres.view(b * n_grids, c, h, w)
    
    # 4. エンコードの実行
    encoded_features = vision_encoder(anyres_input)
    
    # Patch数 + cls_token = (224/16)^2 + 1 = 196 + 1 = 197
    expected_num_patches = (image_size // patch_size) ** 2 + 1
    print(f"Encoded features shape: {encoded_features.shape}")
    assert encoded_features.shape == (b * n_grids, expected_num_patches, vision_dim), "Vision Encoder shape mismatch!"
    
    # 5. LLM次元への射影
    projected_features = vision_projector(encoded_features)
    print(f"Projected features shape: {projected_features.shape}")
    assert projected_features.shape == (b * n_grids, expected_num_patches, llm_dim), "Vision Projector shape mismatch!"
    
    print("✅ All shape assertions passed successfully!")
    print("✅ The advanced vision components (AnyRes & SwiGLU) are ready.")
