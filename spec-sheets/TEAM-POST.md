# 📋 535106 LLM Final — 團隊分工 & 時程（Discord 可貼版）

> 可直接複製貼到團隊群。詳細規格書見 repo `spec-sheets/`。

**題目（交出去的公開版）**：Evaluating Published Faithfulness-Verification Techniques for a RAG QA System
→ **這是 evaluation study**：比較「已發表的」faithfulness 技術、給採用建議。🔴 報告/demo 只講「評估、比較、建議」，**不宣稱自創新方法**。

**兩個核心交付**：① 同一 RAG 上公平比較已發表技術 ② 可重現的採用建議（固定 seed + 釋出 configs）。

---

**分工（4 人）**
```
Emma(414551035) · Verification-layer 整合
  → claim 切分 + citation 對齊 + 接已發表 scorer(embedding/NLI/HW1) + metric harness + 門檻校準

曾俊耀(315551125) · RAG base + baselines
  → Program1/2(向量庫+cited QA) + AnswerBundle + 跑 RAGAS/NLI/SelfCheckGPT baselines

許伯宇(315511057) · Analysis + Writing
  → acc/cost/reproducibility 三軸比較 + 採用建議 + 圖表 + final report

Noa(315554801) · Survey + Data
  → 已發表技術綜述 + RAGTruth/ALCE 策展 + 凍結校準子集 + 可重現釋出
```

**關鍵路徑（誰等誰）**
```
Noa 凍結子集 → Emma 校準 → 跑數字 → 許伯宇分析+圖表 → 投影片
曾俊耀 AnswerBundle(先丟5筆給Emma) + baselines 並行
```
🔴 **最卡 = Noa 凍結子集 + Emma 校準（已逾期 11/17）→ 本週(11/20–22)先清這兩步。**

---

**時程（倒數 → 12/04）**
```
11/20–24  做完實作:曾俊耀 AnswerBundle+5筆example、Noa 凍結子集、Emma Block3-7
11/24     Emma 門檻校準(補逾期)
11/25–26  全組跑完整實驗 → 主結果表
11/27(五) ⚠ W12 Demo:HW1/HW2 notebook 口試(脫稿解釋 + 會問 proposal)
11/28–29  許伯宇:WARN 分析 + 三軸圖 + 結果表
11/30–12/1 可重現釋出 + report 收口
12/02     做投影片
12/03     彩排 1 次
12/04(五) ★ Final Presentation
```

**11/27 Demo 怎麼準備**：助教在他電腦開你繳到 E3 的 notebook，固定問 4 題（①基本觀念 ②code 理解 ③實驗分析）+ Final Proposal。練到能**脫稿解釋自己每個設計選擇**，不背公式。3 位助教各 1–5 分取平均。
