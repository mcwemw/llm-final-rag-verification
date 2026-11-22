# 四人分工規格書（Spec Sheets）— 535106 LLM Final Project
## ✅ 對齊「公開 evaluation 版」（2026-11-20 定案）

> 依**實際交出去的公開 proposal** §6 分工 + `BLOCK-SPECS.md` 工程契約整理。
> 每份 = 一個成員的「設計原理 → 完成順序步驟 → 範例結果 → 交付檢核」。

## 題目（權威 = 交出去的公開版）
**"Evaluating Published Faithfulness-Verification Techniques for a Retrieval-Augmented QA System"**
- **定位 = evaluation study**（Track 2 Dataset/Benchmark + 3 Application）：已**公開**的 faithfulness-verification 技術能否當 RAG 的驗證層？哪個值得採用、trade-off 為何。
- 🔴 **沒有「我方 novel 方法」**（註腳白紙黑字聲明）。報告/demo 只能講「**評估、整合、比較已發表技術 + 給採用建議**」。
- **determinism = 一個被量測的評測軸**（embedding/NLI variance=0、LLM-judge>0），**不是**作者的新發明——當作「量測到的一個性質」講，合規。

## 兩個 load-bearing 交付（公開版 §6 結尾）
1. **公平比較**：在同一個 RAG 系統上，matched-budget 比較已發表的 verification 技術。
2. **可重現的採用建議**：哪個技術值得採用、在什麼 trade-off 下（固定 seed、釋出 splits/configs）。

## 四人分工（對齊公開版 §6，權威）
| 代號 | 角色（公開版用字）| 一句話交付 |
|---|---|---|
| **吳-Gate** | Verification-layer 整合 | HW1 語言ID前濾 + claim 切分 + citation 對齊 + 把**已發表 scorer**(embedding/NLI/HW1 self-attention)接到共同 harness |
| **曾-RAG** | RAG base + baselines | 架 Program1/2 + 載資料集 + 跑 LLM-judge(RAGAS)與既有 baselines |
| **許-Analysis** | Analysis + Writing | accuracy/cost/reproducibility 比較 + **採用建議** + 圖表 + final report |
| **艾-Survey** | Survey + Data | 已發表技術綜述(RAGAS/FActScore/NLI/VeriScore/RAGTruth/Face4RAG) + 語料策展 + 可重現釋出 |

## 檔案
- [`01-gate.md`](01-emma-gate.md) — M-Gate / verification-layer 整合
- [`02-rag-baselines.md`](02-tseng-rag-baselines.md) — M-RAG / RAG base + baselines
- [`03-analysis-writing.md`](03-xu-analysis-writing.md) — M-Analysis / analysis + 採用建議 + writing
- [`04-survey-data.md`](04-noa-survey-data.md) — M-Survey / survey + data
- [`../12-04-presentation-timeline.md`](../12-04-presentation-timeline.md) — 12/4 倒排時程

## 整合點（誰等誰）
```
艾-Survey(綜述+凍結子集) ──┐
曾-RAG(Program1-2 AnswerBundle)──→ 吳-Gate(整合層:切分/對齊/接 scorer/harness) ──→ 許-Analysis(比較+採用建議+report)
                                         ↑
                           曾-RAG(已發表 baselines 同 interface)
```
🔴 **契約先定死再並行**：`schemas/claim_citation.schema.json`、`schemas/scorer_return.schema.json`（吳-Gate 已定稿）是跨人交界，break 要 bump version + review。

## ⚠️ 合規紅線（公開版必守）
- 報告/投影片/demo **不得宣稱任何方法是作者 novel 貢獻**（PASS/WARN/FAIL/UNSOURCED 判定與 determinism 都講成「評估框架 / 被量測的性質」，不是「我們發明的」）。
- 私有 novel 版（`../proposal.tex` "Trustworthy RAG: A Deterministic… Gate"）**不進公開場合**，只留研究線自用。
