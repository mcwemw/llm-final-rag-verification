# score-export-v1 — Scorer 分數交換契約 / Score Export Contract
> 用途:Jimmy 用自己的 Groq key 跑 RAGAS / SelfCheckGPT(或任何 scorer)→ 匯出每個 response 的分數 → harness 讀入算指標(Jimmy #4 option b)。
> 原則:**每份 export 自我描述、可追溯、分數語義固定**。harness 讀取時會 **assert** 下列必填欄位與 `score_direction`,缺則拒收(fail-loud,不猜)。

---

## 1. 一份 export = 一個 (technique × model × run) 在一個 dataset 版本上的結果

```json
{
  "schema_version": "score-export-v1",
  "run_id": "2026-11-27T01-30_selfcheck_gpt-oss",
  "technique": "selfcheck",
  "model": "deberta-v3-large",
  "scorer_version": "baselines.py@<git-sha-or-tag>",
  "dataset_version": "calib_v1",
  "dataset_sha": "<manifest.json 的 sha 或 git sha>",
  "n_samples": 3,
  "generator_model": "openai/gpt-oss-20b",
  "temperature": 1.0,
  "score_direction": "high_means_faithful",
  "score_range": [0.0, 1.0],
  "rerun": false,
  "samples_reset": true,
  "created_at": "2026-11-27T01:30:00+08:00",
  "scores": [
    { "bundle_id": "bundle_03", "claim_id": "claim_001", "score": 0.21, "error": null },
    { "bundle_id": "bundle_03", "claim_id": "claim_002", "score": 0.88, "error": null },
    { "bundle_id": "bundle_07", "claim_id": "claim_001", "score": null, "error": "groq_timeout" }
  ]
}
```

---

## 2. 欄位定義(harness 必檢)

| 欄位 | 必填 | 說明 |
|---|---|---|
| `schema_version` | ✅ | 固定 `"score-export-v1"`;版本不符 harness 拒收。 |
| `run_id` | ✅ | 唯一識別這次執行(時間+technique+model)。同 claim 多次跑用不同 run_id 區分。 |
| `technique` | ✅ | `presence` / `ragas` / `selfcheck` / `nli` / `embedding` …(= harness 比較表的列名來源)。 |
| `model` | ✅ | 判斷用的模型(RAGAS judge model / SelfCheck 的 NLI model)。**不同模型跑的分數不可混為一列**。 |
| `scorer_version` | ✅ | scorer 程式版本(git sha 或 tag),確保可重現。 |
| `dataset_version` | ✅ | 分數跑在哪份資料(如 `calib_v1`);配 `dataset_sha` 防「同名不同內容」。 |
| `dataset_sha` | ⬜ 建議 | 資料的 hash / git sha,鎖死資料內容。 |
| `n_samples` | selfcheck 必填 | 每 response 的抽樣數(sampling consistency 法)。 |
| `generator_model` | selfcheck 必填 | SelfCheckGPT 重抽樣本用的生成模型(≠ 判斷用 NLI model)。 |
| `temperature` | ⬜ | 生成/判斷溫度(RAGAS judge 建議 0;selfcheck 取樣需 >0)。 |
| **`score_direction`** | ✅ **最關鍵** | `"high_means_faithful"` 或 `"high_means_hallucinated"`。**RAGAS 與 SelfCheckGPT 原始分數方向不一定相同**;harness 依此欄統一轉成內部「hallucinated=1」前,不得混算。缺此欄 → 拒收。 |
| `score_range` | ✅ | `[min,max]`,如 `[0,1]`;harness 據此正規化。 |
| `rerun` | ✅ | 是否為重跑(算 determinism 要知道)。 |
| `samples_reset` | selfcheck 必填 | rerun 前有沒有呼叫 `reset_samples()`。**若為 false,Det-var 會被假裝成 0**(Jimmy 已提醒)→ harness 看到 `rerun=true & samples_reset=false` 會警告該次 determinism 不可信。 |
| `created_at` | ✅ | ISO8601 時戳。 |
| `scores[]` | ✅ | 每筆:`bundle_id` + `claim_id` + `score`(float 或 null)+ `error`(字串或 null)。 |

---

## 3. 語義硬規則(harness 不猜,違反即拒收)
1. **方向必宣告**:沒有 `score_direction` 的 export 一律拒收。harness 內部統一成「hallucinated 機率」後才進指標,絕不假設方向。
2. **分數不跨 model 混算**:同一列比較只能來自同 `technique`+`model`;不同 model 各佔一列。
3. **error 不吞**:生成/判斷失敗的 claim `score=null`+填 `error`;harness 把 null 計入「覆蓋率」而非當 0 分(避免退役模型 404 那種「全 0 假裝有跑」)。
4. **determinism 誠實**:Det-var 只在 `rerun=true` 且(selfcheck 時)`samples_reset=true` 的多次 export 間計算;否則標「determinism 未量」。
5. **資料版本對齊**:`dataset_version`/`dataset_sha` 要對得上 harness 當下用的 calib/test split,否則拒收(防「分數跑在舊資料上」)。

---

## 4. harness 端讀取流程(示意)
```
load export.json
assert schema_version == "score-export-v1"
assert all required fields present       # 缺 → raise, 不猜
flip = (score_direction == "high_means_faithful")   # 統一成 hallucinated 機率
for s in scores:
    p_hallu = (max - s)/(max-min) if flip else (s-min)/(max-min)
    ... threshold → verdict → 對 gold 算 P/R/F1/κ
coverage = #(score != null) / #claims         # error 列算覆蓋率，不當 0
```
