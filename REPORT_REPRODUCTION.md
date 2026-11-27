# 報告第柒節 程式碼與重現

公開程式與重現說明固定於版本 reproduction-20261127：https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/tree/reproduction-20261127。本節與 GitHub 的 REPRODUCE.md 共用 config/reproduction_contract.json；模型清單與數位指紋另見 config/public_weights.json。

所需的 27 個權重、分段封存檔及授權檔均可匿名公開下載：https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/releases/tag/trained-weights-20261127。內容包含已微調身分編碼器及其母模型 LM1–LM6、SVD／IDF 與 MLP、段位模型、HumanSL、策略 MLP 及擬合後 LDA；採保存權重進行推論，不需要重新訓練。cx40 必須使用 epoch index 2，SHA256 為 6d44cc780541c286720e715645088b9695bde485ca8861516ea4ab6096040749。

主辦單位把八份原版／新版公開及私有候選庫與測試題 CSV，以及 sample_submission_player_v2.csv 放入 $GO_ROOT/data/tests1v2/tests。執行 --tests-only 檢查，不需要訓練資料。先依相同清單下載並驗證權重，再恢復至新工作目錄；完整命令與母模型路徑見表三與版本化說明。KataGo 來源固定於 d91ea855110dae533f0aada947b2b7d78cc8a4e1。

驗證範圍必須區分：27 項公開資產的匿名存取與檔案雜湊已通過核對；先前保存特徵／分數重播可得到最佳 CSV，但不等同原始資料推論。原始測試資料目前不在本地；歷史父代流程仍有未整合的設定／中間輸入，且 base-view 重建曾改變部分低排名候選。因此尚無已認證可由原始 CSV 一次產生最佳 CSV 的完整命令。表三的元件推論命令不宣稱端到端通過，亦不以複製最佳 CSV 代替推論。

## 與報告相同的命令

```bash
python tools/download_weights.py --check-public
python tools/download_weights.py --directory "$DOWNLOADS"
python tools/restore_weights.py --directory "$DOWNLOADS" --root "$GO_ROOT"
python tools/check_official_data.py --root "$GO_ROOT" --inventory config/official_input_hashes.json --tests-only
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_infer_ablation_1118.py" --root "$GO_ROOT" --tag cx40_lm4_noaug_all
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_policy_test_features_1119.py" --root "$GO_ROOT" --pool pub2
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_policy_test_features_1119.py" --root "$GO_ROOT" --pool priv2
```

最佳輸出 SHA256：`eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26`
