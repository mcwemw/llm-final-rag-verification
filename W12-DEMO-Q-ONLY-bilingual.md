# W12 Demo 口試題目卡 — 第三組 / W12 Demo Oral Exam Questions — Group 3
> 11/27（五 Fri）｜自我練習用 Self-practice — 答完對照 W12-DEMO-QA-bilingual.md

---

## 全組共同 / All Members

1. 你們做了什麼新東西？貢獻是什麼？ / What is your research contribution? Did you invent something new?
2. bge-m3 F1 才 0.31，這樣有用嗎？ / bge-m3 F1 is only 0.31 — is it useful?
3. Determinism 是你們發明的嗎？ / Did you invent "determinism"?

---

## 🟦 吳-Gate（Emma，Block 3–7）

### ① 基本觀念 / Concepts
1. Faithfulness 是什麼？跟 citation presence 有什麼差？ / What is faithfulness? How does it differ from citation presence?
2. Cosine similarity 為什麼不等於 entailment？ / Why is cosine similarity not the same as entailment?
3. WARN 帶的作用是什麼？為什麼不直接 PASS 或 FAIL？ / What is the WARN band for? Why not force binary judgment?
4. Determinism 是什麼？ / What is determinism?

### ② Code 理解 / Code Logic
5. 你的 Block 3–7 整體流程是什麼？（從 AnswerBundle 到 F1 table）/ Walk through your Block 3–7 pipeline from AnswerBundle to the F1 table.
6. Block 4 的三條路徑（explicit / nearest / none）各代表什麼？ / What do the three alignment paths mean?
7. Block 6 的 verdict 規則怎麼寫的？ / How are verdicts assigned in Block 6?
8. HW1 self-attention 是怎麼接進來的？ / How is the HW1 self-attention integrated?

### ③ 實驗分析 / Experimental Analysis
9. Gate bge-m3 的 F1 / Recall / κ 各是多少？ / What are Gate bge-m3's F1 / Recall / κ?
10. 為什麼 Gate F1 低但 recall 高？ / Why does Gate have low F1 but high recall?
11. LLM-judge 和 Gate 的 F1 / Det-var / Cost 各是多少？哪個比較好？ / Compare LLM-judge and Gate on F1 / Det-var / Cost. Which is better?
12. 門檻怎麼校準的？lo / hi 各是多少？ / How were thresholds calibrated? What are lo and hi?

### ④ Proposal Q&A
13. 研究貢獻是什麼（不能說發明新方法）？ / What is the contribution (without claiming a new method)?
14. 為什麼不直接用 RAGAS 就好？ / Why not just use RAGAS?
15. WARN 帶幻覺率多少？說明什麼？ / What is the WARN band hallucination rate? What does it imply?

---

## 🟩 曾-RAG（Block 1–2 + 4 baselines）

### ① 基本觀念 / Concepts
1. RAG 是什麼？解決什麼問題？ / What is RAG? What problem does it solve?
2. AnswerBundle 是什麼？有什麼不變式？ / What is AnswerBundle? What invariants must it satisfy?
3. 為什麼 RAGAS 當 baseline 而不是主方法？ / Why is RAGAS a baseline rather than the main method?

### ② Code 理解 / Code Logic
4. Program 1 和 Program 2 分別做什麼？ / What do Program 1 and Program 2 each do?
5. 四個 baseline 分別是哪四個？各自的特性？ / Name the four baselines and describe each.
6. 什麼叫 uniform interface？為什麼重要？ / What is a uniform interface? Why does it matter?

### ③ 實驗分析 / Experimental Analysis
7. RAGAS Det-var 是多少？說明什麼？ / What is RAGAS Det-var? What does it imply?
8. Presence-only 的作用是什麼？ / What is presence-only used for?

### ④ Proposal Q&A
9. 為什麼不重造 RAG？ / Why not rebuild RAG from scratch?
10. 生成層（Block 1）為什麼刻意隔離在 gate 之外？ / Why is the generation layer deliberately isolated from the gate?

---

## 🟨 許-Analysis（比較分析＋採用建議 / Analysis + Recommendation）

### ① 基本觀念 / Concepts
1. 三軸 trade-off 是哪三軸？ / What are the three trade-off axes?
2. 採用建議怎麼用？成本敏感選什麼？追最高召回選什麼？ / How should the adoption recommendation be applied?

### ② Code 理解 / Code Logic
3. 分析層的工作流程是什麼？ / What is the analysis layer's workflow?
4. WARN 分析怎麼做？三分類是什麼？ / How is WARN analysis done? What are the three categories?

### ③ 實驗分析 / Experimental Analysis
5. 三個方法的 F1 / Det-var / Cost 各是多少？ / What are F1 / Det-var / Cost for each method?
6. 三軸圖怎麼解讀？為什麼沒有單一贏家？ / How do you read the three-axis plot? Why is there no single winner?
7. WARN 帶幻覺率多少？說明什麼？ / What is the WARN band hallucination rate? What does it imply?

### ④ Proposal Q&A
8. 研究不宣稱 novel method，要怎麼說貢獻？ / How do you articulate contribution without claiming novelty?
9. 採用建議的結論句怎麼說？ / What is the adoption recommendation in one sentence?

---

## 🟧 艾-Survey（RAGTruth ground truth＋文獻 / Survey + Data）

### ① 基本觀念 / Concepts
1. 為什麼 gold 不能是 LLM？ / Why can't gold labels come from an LLM?
2. RAGTruth 是什麼？授權是什麼？ / What is RAGTruth? What is its license?
3. Span → claim 對齊是什麼？ / What is span-to-claim alignment?

### ② Code 理解 / Code Logic
4. 你的資料前處理流程是什麼？ / Walk through your data preprocessing pipeline.
5. 凍結校準子集的作用是什麼？ / What is the purpose of the frozen calibration subset?

### ③ 實驗分析 / Experimental Analysis
6. 測試集和校準子集的幻覺率各是多少？為什麼差這麼多？ / What are hallucination rates for test set and calibration subset? Why the difference?
7. 固定的 SEED 是多少？ / What is the fixed seed?

### ④ Proposal Q&A
8. 文獻回顧收斂成一句話？ / Summarize the literature review in one sentence.
9. 凍結子集為什麼不能再動？後果是什麼？ / Why can't the frozen subset be changed? What would happen?
