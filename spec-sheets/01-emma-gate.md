# 分工規格書 ① — 吳美真 Emma（414551035）
## 角色：Verification-Layer 整合（公開 eval 版）

**負責 Block**：3 Claim Segmentation · 4 Claim↔Source Alignment · 5 Grounding Scorer 整合（把**已發表 scorer** 接到統一 interface）· 6 Verdict 判定框架 · 7 Metric Harness。
**一句話（公開版 §6 用字）**：verification-layer integration —— HW1 語言 ID 前濾 + claim 切分 + citation 對齊 + 把**已發表的 scorer**（embedding-similarity、NLI、課程 HW1 self-attention）接到**共同評測 harness**，量測它們的 P/R/F1/κ/determinism/cost。
🔴 **合規**：你做的是「整合+評測已發表技術」，**不是**發明新方法。判定框架與 determinism 都是**評測工具/被量測性質**，不宣稱 novel。

---

## 設計原理（為什麼這樣做）
1. **grounded citation ≠ grounded claim**：蔡炎龍 `citation_note` 只驗 presence（編號在不在），他自承「不驗證每句是否真有證據支持」。本評測層補查 **faithfulness**（引得對不對）——這是「評估既有技術能不能補這格」，不是提新方法。
2. **統一 interface 才能公平比較**：embedding / NLI / HW1 self-attention / LLM-judge 都吃同一 `AlignedClaim`、吐同一 `ScorerResult`。**公平比較是本研究的 load-bearing 交付①**，不是工程巧合。
3. **determinism 是被量測的評測軸**：同輸入重跑，embedding/NLI 的 verdict variance=0、LLM-judge>0——這是我們**量出來報告**的一個性質（§3 metric、§5 reporting 欄），用來支撐「哪個技術值得採用」的建議，**不當作者發明**。
4. **WARN 不確定帶 = 誠實承認 cosine≠entailment**：評測時低信心交人工，而非硬二分。這是評測框架的一部分。
5. **HW1 複用**：P1 LID 當語言前濾、P2 from-scratch self-attention 當其中一個被評測的 scorer，接完 HW1 ladder。

---

## 完成順序步驟（含目標日 / 依賴）
> 🔴 今天=11/20，下列 10/24–11/17 應已完成，請逐項核對「真的做完+測過」再往下。

1. **[已應完成 10/24] 鎖 schema**：`claim_citation.schema.json` + `scorer_return.schema.json` 定稿（跨人交界先死）。
2. **[Block 3] Claim Segmentation**：句界+連接詞+分號/條列的決定性切分。
   - 驗收：≥5 單元測，同輸入 3 次 `Claim[]` 一致。
3. **[Block 4] Claim↔Source Alignment**：claim 落在哪個 citation 字元範圍→`explicit`；無→`nearest` 或 `none`（→UNSOURCED）。
   - 驗收：explicit/nearest/none 三路徑各有測例。
4. **[Block 5] 接已發表 scorer（統一 interface `scorer.score(claim, cited_docs)->ScorerResult`）**：
   - embedding-similarity（bge-m3，本機 CPU、normalize、batch16）
   - NLI entailment（cross-encoder，曾俊耀端或你端接）
   - 課程 HW1 self-attention 相似度 regressor
   - 驗收：embedding scorer 同輸入 3 次 score variance=0。
5. **[Block 6, 校準 11/17→凍結 11/29] Verdict 判定框架**：在 Noa 凍結子集上 per-scorer 校準 `(lo,hi)`（max-F1/PR），校準後凍結寫檔。
   - 規則：`alignment==none→UNSOURCED；score≥hi→PASS；lo≤score<hi→WARN；score<lo→FAIL`。
   - 🔴 門檻不可沿用別模型數字（各 encoder 各自重跑曲線）。
6. **[Block 7, harness 11/24] Metric Harness**：verdict→binary 對 gold 算 P/R/F1/κ；determinism=重跑 3 次 F1 variance；cost=API 計費加總。**每個被評測技術填一列**。
   - 驗收：reporting table 無 TBD；`reproduce.sh` clone 得相同數字。

---

## 範例結果（「做完長這樣」）
**輸入**（曾俊耀 Block2 `AnswerBundle`）：
```json
{ "question": "勞工加班費如何計算？",
  "answer": "平日延長工時前兩小時加給三分之一。雇主得片面取消特休。",
  "citations": [{"doc_id":"labor_law_24","span_start":0,"span_end":40,"text":"...前兩小時加給三分之一..."}] }
```
**評測層輸出（示意）**：
```
claim#1 "平日延長工時前兩小時加給三分之一"  bge-m3 cos=0.74  alignment=explicit → PASS
claim#2 "雇主得片面取消特休"                bge-m3 cos=0.21  alignment=none      → UNSOURCED
summary: PASS 1 / UNSOURCED 1 → 需人工複核
```
**跨技術比較表（load-bearing 交付①，示意，數字待實測）**：
```
technique(published)   F1     κ      determ_var   cost/1k claims
Presence-only [Gao'23] TBD    TBD    0            ≈0
LLM-judge RAGAS [Es'24]TBD    TBD    >0 (measured) API
NLI-based [Honov.'22]  TBD    TBD    0            GPU
Embedding bge-m3       TBD    TBD    0            no judge calls
Self-attn (HW1)        TBD    TBD    0            no judge calls
```
→ 這張表就是「哪個已發表技術值得當 RAG 驗證層」的證據基礎。

---

## 交付檢核（DoD）
- [ ] Block 3–6 各有單元測 + 四類判定邊界測例。
- [ ] embedding/NLI scorer 同輸入 3 次 variance=0（log 存證）。
- [ ] 門檻校準寫進檔案（校準集+數值），非手調。
- [ ] reporting table 每個被評測技術無 TBD，`reproduce.sh` 可重現。
- [ ] 能脫稿講「我們評估/整合這些已發表技術」（demo 會問，**不要說成自己發明**）。

## 依賴
- **上游**：曾俊耀 Block1-2（`AnswerBundle`）、Noa（凍結校準子集 + gold verdict）。
- **下游**：許伯宇（吃你的比較表做 accuracy/cost/reproducibility 分析 + 採用建議）。
