# 分工規格書 ③ — 許-Analysis
## 角色：Analysis + Adoption Recommendation + Writing（公開 eval 版）

**負責**：accuracy / cost / reproducibility 三軸比較 · **採用建議**（哪個已發表技術值得當 RAG 驗證層）· 圖表 · final report 主筆。
**一句話（公開版 §6 用字）**：把各已發表技術的量測結果變成「看得懂、站得住、可重現」的比較與**採用建議**——這是本研究的 load-bearing 交付②。

---

## 設計原理（為什麼這樣做）
1. **本研究的產出是「建議」不是「新方法」**：研究問題 = 「哪個**已發表**技術值得採用、trade-off 為何」。本模組的分析要收斂成一條**可操作建議**（例：「低成本高可重現場景選 embedding/NLI；要最高召回但容忍成本選 LLM-judge」）。
2. **三軸 trade-off 要被看見**：accuracy（F1/κ vs 人工）× cost（$/1k claims、latency）× reproducibility（determinism variance）。用圖把三軸攤開，讀者一眼看出沒有單一贏家 → 這正是「值得評估」的理由。
3. **determinism 當「被量測的性質」寫**：embedding/NLI variance=0 vs LLM-judge>0 是**我們量到的結果**，用來支撐建議，**不寫成作者發明**（合規）。
4. **可重現 = 建議可信的工程證明**：固定 seed、釋出 splits/configs、`reproduce.sh` clone 得相同數字——報告把它當一等公民。

---

## 完成順序步驟（含目標日 / 依賴）
> 本模組的工作大多在 吳-Gate/曾-RAG 出數字之後（11/24 harness 出表起），前期先搭骨架。

1. **[前期，現在就能做] report 骨架**：NeurIPS 6 節對齊公開 proposal（Overview/Formulation/Evaluation/Methodology/Experimental Results/Contributions），搬 proposal 內容進架構、佔好圖表位。
2. **[11/24 後] 主比較表**：各已發表技術 × {F1, κ, determ_var, cost, latency} 排版；標最佳、determ_var=0 粗體。
3. **[11/24 後] 三軸 trade-off 圖**：accuracy–cost–reproducibility（可用散點或雷達），顯示沒有單一技術全贏。
4. **[11/29 門檻凍結後] WARN 帶錯誤分析**：抽 WARN N 筆人工檢視 → 分「真·脈絡外 / 其實該 FAIL / 其實該 PASS」三類統計 → 證明評測框架的 WARN 帶有意義。
5. **[收尾] 採用建議段**：依三軸給明確場景化建議（這是 §6 交付②的核心句）。
6. **[收尾] 可重現釋出**：和 Noa 合併 splits/seeds/configs + `reproduce.sh` 說明進附錄。
7. **[收尾] 誠實限制欄**：句切啟發式、cosine≠entailment、一句多引取 max——照 gate README「限制」欄寫進 report。

---

## 範例結果（「做完長這樣」）
**採用建議句（示意，數字待實測）**：
> 「在 RAGTruth 上，LLM-judge（RAGAS）召回最高（F1 TBD）但每 1k claims 成本 $X 且 variance>0；embedding-similarity（bge-m3）F1 略低但 **variance=0、無 judge 成本、clone 即重現**。**建議**：成本敏感 / 需可重現的部署選 embedding 或 NLI 驗證層；追求最高幻覺召回且容忍成本與不可重現，才用 LLM-judge。」

**WARN 錯誤分析表（示意）**：
```
WARN 帶抽樣 N=30
├─ 真·脈絡外(該 WARN)      18 (60%)  ← 技術命中
├─ 其實該 FAIL(漏擋)         7 (23%)  ← 門檻 lo 可上調
└─ 其實該 PASS(誤降)         5 (17%)  ← 門檻 hi 可下調
```

---

## 交付檢核（DoD）
- [ ] report 6 節齊，每圖表有編號+caption+正文引用。
- [ ] 每個結論句能指回某張表/圖數字（faithfulness 紀律）。
- [ ] 三軸 trade-off 圖 + 主比較表 + WARN 分析表到位。
- [ ] **採用建議**明確、場景化（交付②核心）。
- [ ] 全文無「我方 novel 方法」字眼（合規）；限制欄誠實列。

## 依賴
- **上游**：吳-Gate（比較表、verdict、WARN 樣本）、曾-RAG（baselines 數字）、艾-Survey（splits/configs/survey）。
- **下游**：全組（report + 12/4 投影片的講稿底）。
