# LLAMA_CSTM2 — llama.cpp Custom Patches for GGUF LoRA Training

量子化済み GGUF モデル（Q4_K_M など）に対して、**家庭用 GPU（8〜16 GB VRAM）で LoRA 訓練を可能にする**パッチ群です。

ベースリポジトリ: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

## 修正ファイル一覧

| ファイル | 内容 |
|----------|------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Unified Memory + ピンドホスト RAM フォールバック |
| `ggml/src/ggml-cuda/out-prod.cu` | 全量子化型の GPU バックワード（OUT_PROD 演算） |
| `ggml/src/ggml-opt.cpp` | グラジェントリセット修正（epoch 境界 spike + static graphs 蓄積バグ） |
| `examples/training/finetune.cpp` | n_ctx パディング修正（ゼロロス ubatch バグ修正） |
| `src/llama-context.cpp` | 負ラベルインデックスガード |

---

## パッチの適用方法

llama.cpp の任意バージョンに各ファイルをコピーして上書きします。

```powershell
# llama.cpp リポジトリのルートで実行
Copy-Item "path\to\LLAMA_CSTM2\ggml\src\ggml-cuda\ggml-cuda.cu"  ggml\src\ggml-cuda\ggml-cuda.cu  -Force
Copy-Item "path\to\LLAMA_CSTM2\ggml\src\ggml-cuda\out-prod.cu"    ggml\src\ggml-cuda\out-prod.cu    -Force
Copy-Item "path\to\LLAMA_CSTM2\ggml\src\ggml-opt.cpp"             ggml\src\ggml-opt.cpp              -Force
Copy-Item "path\to\LLAMA_CSTM2\examples\training\finetune.cpp"     examples\training\finetune.cpp     -Force
Copy-Item "path\to\LLAMA_CSTM2\src\llama-context.cpp"             src\llama-context.cpp              -Force
```

---

## ビルド

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF
cmake --build build --config Release --target llama-finetune -j 6
```

> `GGML_CUDA_GRAPHS=OFF` は必須です（訓練グラフとの干渉を防ぐため）。

---

## 訓練データのフォーマット

`User:` / `Assistant:` プレフィックスの SFT 形式（chat template なし）：

```text
User: 質問文
Assistant: 回答文

User: 質問文2
Assistant: 回答文2
```

---

## 基本的な使い方

### LoRA 訓練（新規作成）

```powershell
$env:GGML_CUDA_ENABLE_UNIFIED_MEMORY = "1"   # VRAM 不足時のフォールバックを有効化

.\build\bin\Release\llama-finetune.exe `
    -m  "model.gguf"          `  # ベースモデル（Q4_K_M など）
    -f  "train_data.txt"      `  # 訓練データ（User:/Assistant: 形式）
    --train-sft               `  # SFT モード
    --lora-create             `  # LoRA アダプタを新規作成
    -o  "lora_output.gguf"    `  # 出力ファイル名
    --epochs 1                `
    -lr 1e-4                  `
    --lora-r 4                `  # LoRA rank（4 推奨）
    --ctx-size 256            `  # コンテキスト長（VRAM に応じて調整）
    -ngl 99                   `  # 全レイヤーを GPU にオフロード
    -fa off                   `  # Flash Attention オフ（訓練時は不安定なため）
    2>&1 | Tee-Object "train_log.txt"
```

### 訓練の再開（チェックポイントから）

モデルと同名の `_WORK.gguf` が自動保存されます。  
同じコマンドを再実行するだけで自動的に再開されます。

```powershell
# 同じコマンドを再実行 → _WORK.gguf を自動検出して再開
.\build\bin\Release\llama-finetune.exe -m "model.gguf" --lora-create -o "lora_output.gguf" ...
```

### LoRA を適用して推論

```powershell
.\build\bin\Release\llama-completion.exe `
    -m "model.gguf"             `
    --lora "lora_output.gguf"   `
    -no-cnv                     `  # チャットテンプレートなし
    -p "User: 質問\nAssistant:" `
    -n 200
```

---

## VRAM の目安（Q4_K_M）

| モデル | VRAM 使用量 | 推奨 GPU |
|--------|------------|---------|
| 8B     | ≈ 5 GB     | RTX 3070 8GB 以上 |
| 14B    | ≈ 8 GB     | RTX 3080 10GB 以上 |
| 27B    | ≈ 15 GB    | RTX 5060 Ti 16GB 以上 |

> `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` を設定すると、  
> VRAM が不足した場合にピンドホスト RAM (PCIe 経由) へ自動フォールバックします。  
> 速度は低下しますが、OOM クラッシュを防げます。

---

## ctx-size の選び方

```
ctx-size = GGML_PAD(max_sequence_length, 256) の倍数
```

訓練データの最長シーケンスよりも大きい値を設定してください。  
小さすぎるとデータが切り捨てられ、大きすぎると VRAM を消費します。

| データの最大長 | 推奨 ctx-size |
|--------------|--------------|
| 〜 64 tokens | 128 または 256 |
| 〜 256 tokens | 256 |
| 〜 512 tokens | 512 |
| 〜 1024 tokens | 1024 |

---

## 開発・テスト環境

- GPU: NVIDIA RTX 5060 Ti (VRAM 16 GB, Compute Capability 12.0)
- OS: Windows 11
- CUDA: 12.x
- ベース: llama.cpp (2025年末〜2026年初頭のコミット)
