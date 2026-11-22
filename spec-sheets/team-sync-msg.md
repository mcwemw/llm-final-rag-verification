# 團隊同步訊息草稿（Emma 貼到群組用）

> 2026-11-20。目的:同步時程 + 催關鍵路徑(Noa 凍結子集 / Emma 校準)。語氣友善不逼人。

---

## 版本 A — 完整同步（貼群組）

哈囉大家 🙌 離 12/4 Final Presentation 剩兩週，同步一下進度跟分工：

**題目（交出去的公開版）**：Evaluating Published Faithfulness-Verification Techniques for a RAG QA System
→ 是**評估已發表技術 + 給採用建議**的研究，報告時講「評估/比較/建議」就好，不用說成我們自創方法。

**分工**
・Emma：claim 切分/對齊 + 接 scorer + harness + 門檻校準
・俊耀：RAG base（Program1/2）+ AnswerBundle + baselines(RAGAS/NLI/SelfCheck)
・伯宇：三軸比較 + 採用建議 + 圖表 + report
・Noa：技術綜述 + RAGTruth 策展 + 凍結校準子集

**這週(11/20–22)最關鍵的兩步**（卡住後面全部）：
① **凍結校準子集**（Noa 負責）— Noa 不在群，我另外 email 催，進度我同步給大家；Emma 門檻校準等它開工。
② 俊耀 👉 AnswerBundle 契約鎖好後**先丟 5 筆 example**，Emma 就能端到端測，baselines 可並行補。

**時程**：11/24 Emma 校準 → 11/25-26 跑數字 → 11/27 W12 Demo → 11/28-29 分析+圖表 → 12/2 投影片 → 12/3 彩排 → 12/4 報告。

有卡住的地方隨時喊，我們對一下 🚀

---

## 版本 B — Email 給 Noa（不在群組，走信箱 noa.andre.cs15@nycu.edu.tw）

**To**: noa.andre.cs15@nycu.edu.tw
**Subject**: [535106 Final] Checking in on the frozen calibration subset — needed to unblock threshold tuning

Hi Noa,

Hope you're doing well! Quick sync as we head toward the **Dec 4 final presentation**.

Your part (survey + data) is on the critical path right now — two things block the rest of us:

1. **Frozen RAGTruth calibration subset** — I need this to start the per-scorer threshold tuning (it was slotted for Nov 17, so the sooner the better). Even a first draft of the frozen split + seed would let me start.
2. **span → claim alignment** — mapping RAGTruth's span-level hallucination labels to claim-level gold, so predictions and gold line up. Happy to pair on the format if useful — our schema is in the repo (`schemas/claim_citation.schema.json`).

Could you let me know roughly when the frozen subset can be ready? If anything's blocking you, tell me and we'll sort it out together.

For reference, the full timeline and your spec sheet are here: [repo `deliverables/llm-final-proposal/spec-sheets/04-noa-survey-data.md`].

Thanks so much! 🙏
Emma

---

🔴 **提醒**：以上為草稿，由本人貼/寄（email/群訊需本人確認）。群組訊息(版本A)給俊耀+伯宇；Noa 走 email(版本B)。
