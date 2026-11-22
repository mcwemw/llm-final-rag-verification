# 分工規格書 ④ — 艾-Survey
## 角色：Survey + Data

**負責 Block**：8 RAGTruth 人工標籤（ground truth）· 相關 faithfulness-verification 文獻回顧 · RAGTruth/ALCE 語料策展與前處理 · 釋出 splits/seeds/configs（可重現套件）。
**一句話**：給整個評測一個**非循環、可重現的真相源**——人工標註的 gold，不是第二個 LLM 的意見。

---

## 設計原理（為什麼這樣做）
1. **Ground truth 不能是 LLM**：若用另一顆 LLM 當 gold，就掉進「LLM 評 LLM」的循環性——整個評測的可信度會崩。**用 RAGTruth 的人工 span 標註當 gold，是我們可信度的根**。
2. **固定 split/seed = 可重現的前提**：所有人的數字都要能 clone 重跑。本模組凍結的子集 + SEED 是全組共用的地基，一旦發出去就不能動（動了全組數字作廢）。
3. **文獻回顧要定位「這些技術誰報了什麼」**：不是流水帳，是整理已發表 faithfulness-verification 技術（RAGAS/FActScore/NLI/VeriScore/RAGTruth/FaithBench/Face4RAG）各自的裁判方式、粒度、有沒有報 determinism/cost——**凸顯 determinism 很少被一起量測**，正是本研究把它們放同一 harness 比較的動機。
4. **span→claim 對齊是評測能 claim-level 的關鍵**：RAGTruth 標的是 span，要映射到 吳-Gate Block 4 的 claim 粒度，gold 才對得上 prediction。

---

## 完成順序步驟（含目標日 / 依賴）
> 🔴 今天=11/20，本模組的 11/03–11/12 里程碑應已完成（吳-Gate 的門檻校準 11/17 等本模組的凍結子集）——請確認凍結子集已交付，否則是全組 critical path。

1. **[11/03] 載入 RAGTruth**：`response.jsonl` + `source_info.jsonl`，切 QA split，固定 `SEED=20261021`。
2. **[文獻回顧] faithfulness-verification 綜述**：RAGAS / FActScore / NLI-based / VeriScore / RAGTruth / FaithBench / Face4RAG。
   - 每篇一律附**授權欄（CC 別 + 可否商用）**（組規紀律）。
   - 收斂成一句定位：「這些方法 X/Y/Z，**但都沒把 determinism 當指標**」。
3. **[11/10] span→claim 對齊**：把 RAGTruth 人工幻覺 span 映射到 claim 粒度（配合 吳-Gate Block 4 對齊規則）→ 逐 claim gold verdict。
4. **[11/12] 凍結校準子集**：切出校準子集並**凍結**（吳-Gate 11/17 要在上面選門檻）。發出統計表：#claim、PASS/WARN/FAIL/UNSOURCED 分布。
5. **[收尾] 可重現套件**：打包 splits + seeds + configs，寫 `reproduce.sh`，確保 clone → 得相同數字。

---

## 範例結果（「做完長這樣」）
**凍結子集統計表（示意）**：
```
RAGTruth QA split (SEED=20261021)
#responses: 420   #claims(span→claim 後): 1,187
gold 分布:  PASS 712 / WARN 0* / FAIL 318 / UNSOURCED 157
  *WARN 是我方 gate 的輸出類別,gold 只分 支持/不支持/無來源
校準子集(凍結): 150 claims, 人工覆核雙標一致率 κ=0.82
```
**文獻定位表（示意，全是已發表技術）**：
```
技術(published)  粒度      裁判         有報 determinism?   授權
RAGAS           claim    LLM-judge     ✗                  Apache-2.0(可商用)
NLI (TRUE)      claim    cross-encoder 多半未報            查
RAGTruth        span     人工benchmark  ✗                 CC BY (查)
embedding(bge-m3) claim  cosine        ✗(但本質 variance=0) MIT
```
→ 重點 = **「determinism 很少被一起報」是文獻現況**，本研究把這些技術放同一 harness 補量這一軸（評估，不是發明）。

---

## 交付檢核（DoD）
- [ ] RAGTruth 載入 + QA split + 固定 seed，README 說明 split/seed/schema。
- [ ] span→claim 對齊有測例，`none` 路徑導向 UNSOURCED 與 Emma Block4 一致。
- [ ] 凍結子集**已交付 吳-Gate**（11/17 校準要用）+ 統計表。
- [ ] 文獻每篇附授權欄；綜述收斂出「沒人把 determinism 當指標」這句定位。
- [ ] 可重現套件 clone 得相同數字。

## 依賴
- **上游**：RAGTruth/ALCE 原始語料。
- **下游**：吳-Gate（吃本模組的 gold + 凍結校準子集）、許-Analysis（吃本模組的 splits/configs 做可重現釋出 + survey 進 report related work）。
- 🔴 **本模組卡在校準路徑最前端**：凍結子集沒交，吳-Gate 門檻校準（11/17）就動不了 → 連動 harness（11/24）、report（11/24 後）。**優先確認這件事已完成。**
