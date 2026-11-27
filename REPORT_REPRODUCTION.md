# 報告第柒節 程式碼與重現

公開程式與重現說明固定於版本 inference-20261127：https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/tree/inference-20261127。本節與 GitHub 的 REPRODUCE.md 共用 config/reproduction_contract.json；模型清單與數位指紋見 config/public_weights.json，現行推論程式清單見 config/inference_source_manifest.json。

所需的 27 個公開資產包含身分編碼器及其母模型 LM1–LM6、SVD／IDF 與 MLP、段位模型、HumanSL、策略 MLP 及擬合後 LDA；下載位置為 https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/releases/tag/trained-weights-20261127。使用已訓練權重即可推論，不需要重新訓練。cx40 選用 epoch index 2，SHA256 為 6d44cc780541c286720e715645088b9695bde485ca8861516ea4ab6096040749。

主辦單位把八份候選庫與測試題 CSV，加上 sample_submission_player_v2.csv 放入 $GO_ROOT/data/tests1v2/tests。依版本化說明安裝 requirements/requirements-inference.txt 與 CUDA PyTorch，下載、驗證並恢復權重，固定 KataGo 來源於 d91ea855110dae533f0aada947b2b7d78cc8a4e1。表三提供同一組檢查、執行及續跑命令。

完整推論入口 tools/infer.py 已串接 41 個階段、66 個模型檔案：由 SGF 產生原始分數與段位訊號、身分表示與查詢圖，再重播盤面抽取 HumanSL 特徵、執行 MLP／LDA 高斯匹配及融合，最後依官方範例順序寫出 1600 題。流程不讀取歷史預測 CSV、保存特徵或分數；不執行訓練或 verifier 擬合。輸出為 sub/g23_greplace.csv，並產生逐階段日誌、雜湊及完成紀錄。只有實際檔案符合最佳 SHA256 才回報成功；不一致會保留輸出供診斷並以失敗結束。

檢查範圍如下：27 個公開資產的匿名存取與雜湊通過；新融合程式與歷史演算法的合成測試逐位一致，使用真實保存中間結果時，新版兩池的 800 題前五名均符合接受版。cx40 與 HumanSL 已用合成合法棋譜通過真實 GPU 前向檢查；這些不是原始測試資料的端到端驗證。完整命令已實作，但原始測試資料目前不在本地，且先前 base-view 重建差異仍需在資料恢復後核對，因此尚不宣稱原始資料到最佳 CSV 的雜湊驗證已通過。

## 與報告相同的命令

```bash
python tools/download_weights.py --check-public
python tools/download_weights.py --directory "$DOWNLOADS"
python tools/restore_weights.py --directory "$DOWNLOADS" --root "$GO_ROOT"
python tools/check_official_data.py --root "$GO_ROOT" --inventory config/official_input_hashes.json --tests-only
python tools/infer.py --root "$GO_ROOT" --gpu 0 --preflight-only
python tools/infer.py --root "$GO_ROOT" --gpu 0
python tools/infer.py --root "$GO_ROOT" --gpu 0 --resume
python tools/infer.py --root "$GO_ROOT" --plan
```

最佳輸出 SHA256：`eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26`
