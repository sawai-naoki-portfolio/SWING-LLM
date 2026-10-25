import torch.nn as nn

class SWIVisionLanguageModel(nn.Module):
    """
    MiniLLM (テキストモデル) と VisionEncoder を統合する VLM ラッパーモデル。
    LLM自体の重みを変更せずにマルチモーダル化を実現します。

    NEW-L5: `video_encoder`（core/vision.VideoFrameEncoder）を注入すると、
    `videos=[B, T, C, H, W]` のフレーム列も vision トークンとして注入できる
    （時空間エンコード → VisionProjector 共用 → text_model の vision_features 経路）。
    """
    def __init__(self, text_model, vision_encoder, vision_projector, video_encoder=None):
        super().__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.vision_projector = vision_projector
        self.video_encoder = video_encoder            # None なら従来どおり画像のみ

    @property
    def max_seq_len(self):
        return self.text_model.max_seq_len

    def encode_video(self, videos):
        """フレーム列 [B, T, C, H, W] → LLM 次元の vision トークン [B, T, llm_dim]（NEW-L5）。"""
        if self.video_encoder is None:
            raise ValueError("video_encoder が注入されていません（core/vision.VideoFrameEncoder を渡す）。")
        return self.vision_projector(self.video_encoder(videos))

    def forward(self, input_ids, start_pos=0, use_cache=False, past_key_values=None, images=None,
                videos=None, return_mtp=False, return_prm=False):
        # NEW-L5: 動画（フレーム列）。画像と同じ vision_features 経路で注入する。
        if videos is not None:
            return self.text_model(
                input_ids,
                start_pos=start_pos,
                use_cache=use_cache,
                past_key_values=past_key_values,
                vision_features=self.encode_video(videos),
                return_mtp=return_mtp,
                return_prm=return_prm
            )

        # 画像がない場合は純粋なテキストモデルとして推論
        if images is None:
            return self.text_model(
                input_ids,
                start_pos=start_pos,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_mtp=return_mtp,
                return_prm=return_prm
            )
            
        # 画像の特徴抽出と次元変換
        # L-5: AnyRes 対応。images が [B, num_tiles, C, H, W] の 5D（複数タイル）の場合、
        # 各タイルを個別にエンコードして vision トークンを連結する（MiniLLM.forward は
        # vision_features の系列長 num_vision を可変で扱える）。4D [B,C,H,W] は従来どおり。
        if images.dim() == 5:
            b, n_tiles, c, h, w = images.shape
            feats = self.vision_encoder(images.reshape(b * n_tiles, c, h, w))   # [B*T, P, vdim]
            feats = self.vision_projector(feats)                                # [B*T, P, llm_dim]
            p, d = feats.shape[1], feats.shape[2]
            vision_features_projected = feats.reshape(b, n_tiles * p, d)         # [B, T*P, llm_dim]
        else:
            vision_features = self.vision_encoder(images)
            vision_features_projected = self.vision_projector(vision_features)

        # テキストモデルに画像特徴量を渡して推論
        return self.text_model(
            input_ids, 
            start_pos=start_pos, 
            use_cache=use_cache, 
            past_key_values=past_key_values,
            vision_features=vision_features_projected,
            return_mtp=return_mtp,
            return_prm=return_prm
        )
