# 分工規格書 ② — 曾-RAG
## 角色：RAG Base + Baselines

**負責 Block**：1 RAG QA System · 2 AnswerBundle 產生者 · 5 Grounding Scorer 的 **baselines**（RAGAS / NLI / SelfCheckGPT）· 協力 Block 7 指標 harness。
**一句話**：把課程 Program 1/2 架起來產生「待驗證答案」，再實作四個對照 baseline，讓 吳-Gate 有東西可驗、有對手可比。

---

## 設計原理（為什麼這樣做）
1. **生成層刻意隔離在 gate 之外**：RAG QA 是非決定性生成層（Block 1 ❌ determinism）。把它和驗證層（Block 3–7）解耦，gate 才能宣稱 variance=0——**本模組的產出是「被驗的對象」，不是 gate 的一部分**。
2. **baselines 必須吃同一 interface**：所有被評測技術（含 Emma 接的 embedding/NLI/HW1）都走 `scorer.score(claim, cited_docs)->ScorerResult`。**公平比較是本研究 load-bearing 交付①**——同輸入、同指標，才能說「哪個已發表技術在 F1 / cost / variance 上較優」。
3. **LLM-judge 是被評測技術之一**：RAGAS（groq LLM-judge）的 variance>0、cost>0 是**被量測的性質**，和其他已發表技術一起進比較表，讓採用建議有依據——不是攻擊它，是評估它。
4. **不重造 RAG**：直接用課程原始碼 `yenlung/Generative-AI` 的兩支 notebook，省時且與課程脈絡一致：
   - **Program 1**：`RAG_01_打造你的_RAG_系統.ipynb`（建 embedding index + FAISS 向量庫）
   - **Program 2**：`RAG_02_使用篇.ipynb`（cited QA，走 groq gpt-oss）
   - 語料資料：`rag-data.zip`；預建向量庫：`vector_embedding.zip`

---

## 完成順序步驟（含目標日 / 依賴）
> 🔴 今天=11/20，Block1-2 目標日 10/27–10/30 應已完成，請核對；baselines 若未齊是目前最可能的 critical path。

1. **[Block 1, 10/27] Program 1**（`RAG_01_打造你的_RAG_系統.ipynb`）：建 embedding index（課程向量庫+FAISS），語料=打工權益 KB（`rag-data.zip`）。跑通檢索 top-k。
2. **[Block 1-2, 10/30] Program 2 + AnswerBundle**（`RAG_02_使用篇.ipynb`）：cited QA 生成帶 citation 的答案 → **嚴格輸出 `AnswerBundle` schema**（吳-Gate 10/24 定）。
   - 不變式：每 citation 的 `doc_id` 存在於語料庫、`span_*` 落在 doc 長度內。
   - 驗收：JSON Schema draft2020-12 validate 過；存 **5 筆 example** 進 repo（吳-Gate 開發要吃）。
3. **[Block 5 baselines] 四個對照 scorer（同 interface）**：
   - **presence-only**：Program 2 自帶的「有沒有引」→ 下限對照。
   - **RAGAS**（LLM-judge, groq）：🔴 決定性=否，隔離為對照組，專門展示 variance>0 / cost>0。
   - **NLI-TRUE**（cross-encoder entailment）：決定性、非 LLM，entailment 強於 cosine 的對照。
   - **SelfCheckGPT**：取樣一致性法，另一條 LLM-based 對照。
   - 驗收：四者吃同一 `AlignedClaim`、吐同一 `ScorerResult`（含 latency_ms / cost_usd）。
4. **[Block 7 協力] 指標 harness 骨架**：和 吳-Gate 對齊 `{f1, precision, recall, cohen_kappa, determinism_var, cost_per_claim}` 的計算；cost 計費加總由本模組接 API 計量最準。

---

## 範例結果（「做完長這樣」）
**Program 2 輸出一筆 `AnswerBundle`**（= 上游給 吳-Gate 的契約）：
```json
{ "question": "試用期可以不給勞保嗎？",
  "answer": "試用期仍屬勞動契約，雇主應依法加保勞保。",
  "citations": [{"doc_id":"labor_insurance_11","span_start":12,"span_end":58,"text":"...到職當日應申報加保..."}] }
```
**四 baseline 對同一 claim 的 `ScorerResult`（示意）**：
```
method         score  latency_ms  cost_usd  determ(3×)
presence-only  1.00   0.2         0.000     var=0   (只看有沒有引, 粗)
bge-m3(Emma)   0.73   8.1         0.000     var=0
NLI-TRUE       0.81   15.3        0.000     var=0
RAGAS(judge)   0.90   640         0.003     var>0   ← 對照組:貴且不穩
SelfCheckGPT   0.77   1200        0.006     var>0
```
→ 這張表就是全組 load-bearing 交付①「matched-budget 公平比較」的原料。

---

## 交付檢核（DoD）
- [ ] Program 1/2 端到端跑通，5 筆 example `AnswerBundle` 進 repo 且過 schema validate。
- [ ] 四 baseline 全走統一 interface，吐齊 score/latency/cost。
- [ ] RAGAS/SelfCheckGPT 的 variance>0、cost>0 有實測數字（不是口頭宣稱）。
- [ ] 與 Emma 對齊 harness 的 metric 計算口徑一致。

## 依賴
- **上游**：課程原始碼（`RAG_01_*.ipynb` / `RAG_02_*.ipynb` from `yenlung/Generative-AI`）、艾-Survey 的語料（RAGTruth/ALCE）。
- **下游**：吳-Gate（吃本模組的 `AnswerBundle` + baselines 做統一量測）。
- 🔴 **本模組是目前最可能的瓶頸**：吳-Gate 等本模組的 5 筆 example 才能端到端測。**AnswerBundle 契約一鎖死就先丟 5 筆過去**，baselines 可並行補。
