# API 仕様書 — SWING-LLM

| 項目 | 内容 |
|---|---|
| 対象 | `prototype/web/app.py`・`prototype/web/routes/*.py`（FastAPI） |
| ベース URL | `http://127.0.0.1:5000`（`SWI_WEB_HOST` / `SWI_WEB_PORT` で変更） |
| クライアント | ブラウザの Web UI（`web/static/`）と、モバイルアプリ（Flutter・別リポジトリ）。**同じ API** を使う |
| 関連 | [設計書 6 章（Web API とセキュリティ）](design.md#6-web-api-とセキュリティ)・[シーケンス図①](diagrams/svg/seq-chat.svg)・[シーケンス図②](diagrams/svg/seq-tts-clone.svg) |

---

## 1. 共通仕様

### 1.1 ミドルウェア（外側から順）

| 順 | ミドルウェア | 判定 | 拒否時の応答 |
|---|---|---|---|
| 1 | `SecurityGuardMiddleware` | ① Host が許可リストにあるか（常時。`SWI_ALLOWED_HOSTS="*"` で無効化）<br>② 状態を変える HTTP（GET/HEAD/OPTIONS 以外）と WebSocket は Origin も検証<br>③ `SWI_API_TOKEN` 設定時は `/api/*` と `/ws/*` でトークン必須（OPTIONS は除く） | HTTP: `400 invalid host` / `403 cross-origin request blocked` / `401 unauthorized`（JSON `{"error": ...}`）<br>WS: accept の前に close `1008` |
| 2 | `RateLimitMiddleware` | クラス別のトークンバケット（キーはトークンのハッシュ、無ければ IP）、ボディ上限 12MB（Content-Length） | `429 rate limit exceeded` + `Retry-After` / `413 request body too large`<br>WS: accept の前に close `1013` |
| 3 | `CORSMiddleware` | `allow_origins=["*"]`、`allow_credentials=False` | — |

### 1.2 認証

- 送り方: `Authorization: Bearer <token>` または `X-API-Token: <token>`。WebSocket に限りクエリ `?token=<token>` も受け付ける（ブラウザの WS は任意のヘッダを付けられないため）。
- 比較は `hmac.compare_digest`（定数時間）。
- 既定（トークン未設定・ローカル待受）は認証なし。公開する場合はトークンが必須。

### 1.3 レート制限のクラス（`web/rate_limit.py`・`constants.py`）

| クラス | 対象パス | 1 分あたり | バースト |
|---|---|---|---|
| generate | `/api/generate`, `/api/generate_stream` | 60 | 30 |
| image | `/api/generate_image` | 10 | 4 |
| tts | `/api/tts`, `/api/tts_stream`, `/ws/voice-stream`, `/api/asr` | 60 | 30 |
| default | その他の `/api/*`・`/ws/*` | 300 | 60 |

`SWI_RATE_LIMIT=0` で無効化できる（起動時に WARNING）。`SWI_TRUST_PROXY_HEADERS=1` のときだけ `X-Forwarded-For` を信頼する。

### 1.4 入力の上限

| 項目 | 上限 |
|---|---|
| プロンプト | 4,000 文字（超えたら 400。TTS と画像は切り詰め） |
| 会話履歴 | 新しい方から合計 16,000 文字まで、プロンプト組み立ては 8 ターンまで |
| 画像（base64） | 800 万文字、復号後 2,400 万画素。先頭の 1 枚だけ使う |
| 参照音声・ASR 音声（base64） | 1,200 万文字 |
| フィードバック | query / response それぞれ 4,000 文字、プロセス全体で 30 件/分 |

## 2. エンドポイント一覧

| メソッド | パス | 用途 | 成功時の応答 |
|---|---|---|---|
| POST | `/api/generate_stream` | 対話（ストリーミング） | `text/plain` を逐次送信 |
| POST | `/api/generate` | 対話（同期・ツールなし） | `{"response": str}` |
| GET | `/api/llm/status` | LLM の機能出し分け | `{"backend","model","capabilities"}` |
| POST | `/api/feedback` | 評価の収集（継続学習用） | `{"status": "ok"}` |
| POST | `/api/tts` | 音声合成（一括） | `audio/wav` |
| POST | `/api/tts_stream` | 音声合成（ストリーミング） | `text/plain`（1 行 = base64 の WAV チャンク） |
| WS | `/ws/voice-stream` | 音声合成（双方向・割り込み対応） | バイナリ（WAV チャンク）+ テキスト（JSON） |
| GET | `/api/voice/status` | TTS の可否 | `{"available": bool, "engine": "ar"\|"flow"\|"hybrid"}` |
| POST | `/api/asr` | 音声認識 | `{"text": str}` |
| GET | `/api/asr/status` | ASR の可否 | `{"available": bool, "model": str\|null}` |
| POST | `/api/generate_image` | 画像生成 | `image/png` |
| GET | `/api/image/status` | 画像生成の可否 | `{"available": bool}` |
| GET | `/` ほか | 静的ファイル（Web UI） | HTML / JS / CSS |

## 3. 各エンドポイントの詳細

### 3.1 `POST /api/generate_stream`（主経路）

**リクエスト**（`GenerateRequest`）

| フィールド | 型 | 既定 | 説明 |
|---|---|---|---|
| `prompt` | str | `""` | ユーザー発話。`prompt` と `images` の両方が空なら 400 |
| `use_cot` | bool | `false` | `true` で思考過程（`<think>`）つき生成 |
| `images` | str[] | `[]` | base64（data URL 可）。先頭 1 枚のみ使用。プロンプトが空なら既定の画像説明プロンプト |
| `persona` | str | `"jarvis"` | `"jarvis"` / `"navi"` |
| `client` | str | `"web"` | `"flutter"` ならツール通知をプレーンテキストで返す |
| `best_of_n` | int | `0` | このエンドポイントでは使わない（同期版のみ） |
| `history` | `{role, content}[]` | `[]` | 過去の会話。現在の発話は含めない |

**応答**: `200 text/plain`。次の要素が順番に流れる。

| 要素 | 形式 | 意味 |
|---|---|---|
| ツール実行の通知（意図検知時） | Web: `<div class="tool-box success"><span class="tool-check">✔</span><span>✨ 意図を検知しツールを自動実行しました</span></div>`<br>Flutter: `[🔧✨ 意図を検知しツールを自動実行しました]` | ルールベースでツールを先回り実行した |
| ツール結果（意図検知時） | Web: `<tool_result>計算結果: 123*45 = 5535</tool_result>`<br>Flutter: `🔧 計算結果: 123*45 = 5535` | 結果を利用者に直接見せる（最大 600 文字） |
| 本文 | プレーンテキストの差分 | モデルの出力 |
| 思考（CoT 時） | `<think> ... </think>` | 思考過程 |
| ツール呼び出し（ReAct） | `⚙️ [システム: ...]` / `⚠️ [システム: ...]` / `🛑 [システム: ...]` | モデル自身のツール実行の成否 |
| 根拠表示 | `"grounded": true\|false` | RAG の根拠の有無（UI がバッジに置換） |
| エラー | 本文が `Error: ...` で始まる、または末尾に `\n[ERROR] ...` | 失敗（画像の解析失敗など） |

**エラー**: `400`（空の入力、プロンプト長の超過）、`401` / `403` / `400`（境界防御）、`413`、`429`。

### 3.2 `POST /api/generate`

リクエストは 3.1 と同じ。`best_of_n > 1` なら PRM による Best-of-N を使う（0 なら `INFERENCE_CONFIG.best_of_n`）。`use_cot=true` なら思考つきで生成する。

- 応答: `{"response": "..."}`
- エラー: `400`（入力や画像が不正）、`500 {"error": str}`
- **ツールは動かない**（既知の課題。[設計書 9 章](design.md#9-既知の課題と今後)）。

### 3.3 `GET /api/llm/status`

```json
{"backend": "local", "model": "SWI-LLM (MiniLLM / from-scratch)",
 "capabilities": {"vision": true, "cot": true, "best_of_n": true, "tools": true}}
```

外部の LLM ゲートウェイは撤去済みなので、常に `local` を返す。キーの構成はクライアントとの互換のために維持している（テスト `test_llm_status_endpoint.py`）。

### 3.4 `POST /api/feedback`

- リクエスト: `{"query": str, "response": str, "rating": 1 | -1}`
- 応答: `200 {"status":"ok"}` / `400`（rating が ±1 以外、長すぎる）/ `429`（30 件/分を超えた）/ `507`（保存先の容量上限）/ `500`

### 3.5 音声

| エンドポイント | リクエスト | 応答・エラー |
|---|---|---|
| `POST /api/tts` | `{"text": str, "lang": "ja", "ref_audio": base64?, "ref_speaker_id": str?}` | `200 audio/wav`（`Cache-Control: no-store`）。`400`（text が空、参照音声が大きすぎる・不正）、`403`（クローンの同意なし）、`503`（音声モデルが未学習）、`500` |
| `POST /api/tts_stream` | 同上 | `200 text/plain`（1 行 = base64(WAV)。失敗は `__ERROR__<msg>` の行）。同意の拒否はストリーム開始前に `403` |
| `WS /ws/voice-stream` | 送信（テキスト JSON）: `{"action":"generate","text":"..."}` / `{"action":"stop"}` | 受信: WAV チャンク（バイナリ）を順に送り、最後に `{"status":"done"}`。音声モデルが未学習なら close `1011`。`stop` はそのセッションの生成だけを止める |
| `POST /api/asr` | `{"audio": base64, "lang": "ja"}` | `200 {"text": str}`。`400`（空・大きすぎる・不正・未対応形式）、`503`（ASR が使えない）、`500` |

### 3.6 画像

| エンドポイント | リクエスト | 応答・エラー |
|---|---|---|
| `POST /api/generate_image` | `{"prompt": str, "steps": int}`（0 = 既定の 28） | `200 image/png`（`no-store`）。`400`（prompt が空）、`503`（画像モデルが未学習）、`500` |

## 4. 運用に関わる環境変数（Web）

| 変数 | 既定 | 意味 |
|---|---|---|
| `SWI_WEB_HOST` / `SWI_WEB_PORT` | `127.0.0.1` / `5000` | 待受 |
| `SWI_ALLOWED_HOSTS` | （なし） | Host・Origin の追加の許可リスト（カンマ区切り）。`*` で検証を無効化（非推奨）。**公開バインドでは必須** |
| `SWI_API_TOKEN` | （なし） | 設定すると `/api/*` と `/ws/*` で認証が必須 |
| `SWI_RATE_LIMIT` | `1` | `0` でレート制限を無効化 |
| `SWI_TRUST_PROXY_HEADERS` | `0` | `1` で `X-Forwarded-For` を信頼（リバースプロキシの後ろでのみ） |
| `SWI_ENABLE_CODE_EXEC_TOOL` | `0` | `1` でモデルからの `execute_python` を許可 |
| `SWI_ALLOW_UNSANDBOXED_EXEC` | `0` | Docker が無いときのコード実行の代替を許可（非推奨） |
| `SWI_TTS_ENGINE` | `ar` | `ar` / `flow` / `hybrid` |
| `SWI_ASR` | （有効） | `0` で ASR を無効化 |
