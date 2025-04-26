[425講稿](https://www.notion.so/425-1dfb4b9dc1db80e6b8edf3e078357659?pvs=21)

[0502報告](https://www.notion.so/0502-1e0b4b9dc1db80c599d5e4ba17048089?pvs=21)

**Ref** 
https://www.perplexity.ai/search/llm-vllm-benchmark-review-surv-9CnPlFU.R3e69N5BjlPTVQ

https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/

| Benchmark 名稱 | 測試重點 | 特點與最新進展 | Benchmark 指標 | 簡介說明 |
| --- | --- | --- | --- | --- |
| **IFEval** | 驗證式指令遵循（如字數、關鍵字出現次數） | 由 Google Research 提出，包含 25 種可驗證指令、約 500 個提示，強調重現性與自動化評估工具[arXiv](https://arxiv.org/abs/2311.07911?utm_source=chatgpt.com) | Instruction Following Evaluation | 測量模型對指令的理解與執行能力。 |
| **BBH** | 困難語意與邏輯推理題（BIG‑Bench 中表現最弱的任務） | 包含 23 題先前無模型可超越人類平均水平；Chain‑of‑Thought 提示使 PaLM、Codex 成功突破部分題目；最新 Confident‑AI 文件持續更新實作細節[GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard?utm_source=chatgpt.com)[docs.confident-ai.com](https://docs.confident-ai.com/docs/benchmarks-big-bench-hard?utm_source=chatgpt.com) | Big‑Bench Hard | 針對困難語意與邏輯推理問題的測試集合。 |
| **MATH** | 大學層級數學題（代數、幾何、微積分等） | Meta‑Evaluation 包含 1,084 道解答範例評估 LLM 判分品質；同時出現 FrontierMath 等更高難度新測試集[toloka.ai](https://toloka.ai/math-benchmark?utm_source=chatgpt.com)[Epoch AI](https://epoch.ai/frontiermath?utm_source=chatgpt.com) | MATH | 測試模型在中高階數學題目的解題能力（如代數、幾何等）。 |
| **GPQA** | 研究生級物理與相關科學多選題 | 由領域專家設計 448 題，包括物理、生物、化學；2023 年發表於 arXiv，並收錄於 Papers With Code 平台[arXiv](https://arxiv.org/abs/2311.12022?utm_source=chatgpt.com)[學術與代碼](https://paperswithcode.com/dataset/gpqa?utm_source=chatgpt.com) | Graduate‑level Physics Questions and Answers | 評估物理推理與理解能力。 |
| **MUSR** | 多步軟性推理敘事題（謀殺懸疑等） | 採用 neurosymbolic 合成→自然語言生成算法，題目篇幅約 1,000 字，可挑戰最先進模型[arXiv](https://arxiv.org/abs/2310.16049?utm_source=chatgpt.com) | Multilingual Understanding and Reasoning | 多語言理解與邏輯推理測試。 |
| **MMLU** | 多領域多任務選擇題（57 個科目） | 由 Hendrycks 等人提出，衡量模型預訓練知識；持續更新於 Papers With Code 與 Confident‑AI 平台[學術與代碼](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu?utm_source=chatgpt.com)[docs.confident-ai.com](https://docs.confident-ai.com/docs/benchmarks-mmlu?utm_source=chatgpt.com) | Massive Multitask Language Understanding | 57 項多任務能力總和評量，涵蓋醫學、法律、STEM、社科等。 |
| **CO₂ Cost** | 訓練與推論碳排放估算 | Luccioni 等實證 Hugging Face BLOOM 最終訓練碳排約 11.2 t CO₂ eq；相關報告與工具（如 Cutter Consortium）探討減碳與計價挑戰[Medium](https://nathanbaileyw.medium.com/the-carbon-footprint-of-llms-a-disaster-in-waiting-6fc666235cd0?utm_source=chatgpt.com)[cutter.com](https://www.cutter.com/article/large-language-models-whats-environmental-impact?utm_source=chatgpt.com) | CO₂ Cost | 訓練/推論所產生的碳排放指標，評估模型環境可持續性。 |

| **Benchmark 名稱** | **測試重點** | **特點與最新進展** |
| --- | --- | --- |
| **LiveBench** | 多領域綜合能力（數學、程式、推理等） | • 由 Meta、NYU、Johns Hopkins 等機構合作開發，題目來自最新學術競賽、論文實例與新聞報導，確保不落入模型訓練集範圍內• 每月動態更新題庫，目前頂尖模型整體準確率仍低於 70%，顯示未來持續優化空間 |
| **Humanity’s Last Exam (HLE)** | 高難度跨學科推理與多模態理解 | • 由 Center for AI Safety 與 Scale AI 聯合發起，共收錄 2,500 題，涵蓋數學、物理、醫學、AI 原理等領域• 最新成績顯示 GPT‑4o 領先，以約 20.3% 正確率拔得頭籌，但整體仍在 25% 以下 |
| **SCALAR** | 學術長文閱讀與引用推理能力 | • 利用即將在 ICLR 2025 發表的論文與其引用網絡自動生成測試集，題目篇幅可達 3,000 字• 支援動態更新與全自動評分管線，並已開源部分評測工具以便社群擴展 |
| **DynaCode** | 程式碼生成的複雜度適應能力 | • 由 Google DeepMind 等團隊提出，生成超過 1.89 億個嵌套函式呼叫結構的程式題庫• 涵蓋 4 種難度等級、16 種呼叫圖類型；模型在此基準上的平均正確率僅 45.7%，較傳統程式碼基準低 16.8% |
| **FLEX** | 公平性與偏見的魯棒性測試 | • 測試模型在面對各類誘導偏見（性別、種族、政治傾向等）提示時的回應穩定度• 最新成果引入「混合情境對話」（mixed‑scenario prompts），量化偏見分數並提出修正建議 |
| **AILuminate** | 安全性與風險評估 | • 評估模型對仇恨言論、暴力煽動、自我傷害等敏感內容的檢測與拒絕能力• 根據最新報告，GPT‑4o 與 Llama 系列在「敏感度」與「整體安全性」上均獲「良好（Good）」評級 |
| **MILU** | 印度語系多語言理解能力 | • 包含 11 種印度官方語言、85,000 道多選題，跨越 40 多個主題領域• 最新測試指出 GPT‑4 在 MILU 上達 72% 正確率，顯著領先同級別模型，但仍有部分低資源語言表現可提升 |
| **Benchmark Self‑Evolving** | 多代理動態基準生成 | • 採用多代理強化互動機制自動生成新測試實例，實時擴充題庫並評估模型穩定性• 透過「代理博弈」（agent‑game）方式量化模型間的性能差異，目前已迭代更新超過 10 萬個測試場景 |
| **Grid‑Based Game Benchmark** | c | • 以井字棋、連線四子棋等經典格點遊戲為基礎，結合圖像輸入與多輪策略推理任務• 最新版本加入隨機障礙物配置，測試模型隨機適應與規劃路徑的能力 |

| **基准名称** | **主要领域** | **核心任务类型** | **数据规模** | **评估重点** | **独特功能/技术亮点** |
| --- | --- | --- | --- | --- | --- |
| **SuperGLUE** | 通用语言理解 | 问答/指代消解/推理 | 8个任务集 | 复杂语义推理能力 | 保留GLUE最难的2个任务，新增6个高难度任务 |
| **MultiMedQA** | 医疗领域 | 临床诊断/研究QA/消费者咨询 | 6个医疗数据集 | 事实准确性/伦理合规性 | 首个集成专业考试与消费者问题的医疗评估体系 |
| **FinBen** | 金融领域 | 财报分析/风险评估/股票交易 | 24任务/42数据集 | 决策可靠性/市场预测能力 | 首创股票交易评估模块与RAG增强测试框架 |
| **CMMLU** | 中文多领域 | 常识推理/专业领域知识 | 67个主题12k+题目 | 本土化知识适配性 | 包含驾驶规则等特色中文场景测试 |
| **LV-Eval** | 长文本处理 | 超长上下文信息检索 | 256k tokens长度支持 | 抗干扰/关键信息定位能力 | 采用混淆事实插入与关键词替换技术 |
| **MT Bench** | 多轮对话 | 编程/开放领域对话/专业咨询 | 多轮对话任务集 | 上下文连贯性/逻辑一致性 | AI助手自动评分与位置偏差消除机制 |

| 基準 | 測試重點 | 主要領域 | 核心任務類型 | 資料規模 | 獨特功能／技術亮點 | 評估重點的方式 |
| --- | --- | --- | --- | --- | --- | --- |
| **CO₂ Cost** | 模型推理之碳足跡評估 | NLP 推理效率 | 多基準推理輸出之能源消耗衡量 | 涵蓋 2,742 款模型在多項任務上的推理記錄 [Hugging Face](https://huggingface.co/blog/leaderboard-emissions-analysis?utm_source=chatgpt.com) | 統一硬體與並行策略計算 CO₂ 排放量；按能效比（CO₂／回合）呈現口碑排行 | CO₂ 排放量／推理次數比排名 |
| **LiveBench** | 蛋白質結構預測之迭代評價 | 生物信息學 | 同源建模與二級結構預測 | 每週新釋出之數百至千餘蛋白靶標；累計上千場測試 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2373940/?utm_source=chatgpt.com)[Wikipedia](https://en.wikipedia.org/wiki/LiveBench?utm_source=chatgpt.com) | 持續自動化評估；即時比對新入庫 PDB 結構；多指標（RMSD、MAXSUB、對齊準確率）綜合排名 | 以結構重疊分數（RMSD、MAXSUB）與對齊精度排名 |
| **Humanity’s Last Exam (HLE)** | 多模態前沿學術題庫答題準確度評估 | 學術知識綜合 | 閉式多選與填空題回答 | 2,500 道題，涵蓋 100+ 學科，多模態題型 10% 以上 [AGI Safe](https://agi.safe.ai/) | 私有測試集防止資料洩漏；題庫難度動態更新；結合圖文本題目 | 準確率及跨學科平均分比較 |
| **SCALAR** | 長文本科學文獻之引用邏輯推理評估 | 科學文獻理解 | 引用填空（cloze）式推理 | 基於 ICLR 2025 論文及其引用網絡，動態生成數千條測試 [arXiv](https://arxiv.org/abs/2502.13753) | 無需人工標註；難度可控；動態更新機制防止模型「見過即答」 | 依不同上下文長度與推理類型分場次評估長度影響 |
| **DynaCode** | 動態複雜度感知之程式碼生成評估 | 程式碼生成 | LLM 生成嵌套函式程式碼 | 可生成最高 1.89 億 道唯一程式問題，涵蓋 4 種複雜度單元 × 16 種呼叫圖 [arXiv](https://arxiv.org/abs/2503.10452?utm_source=chatgpt.com) | 動態資料集避免抄襲；複雜度與呼叫圖結合度量；可視化回歸性能降幅隨複雜度上升 | 隨複雜度分級測量平均通過率及性能降幅 |
| **FLEX** | 極端對抗性場景下之公平性魯棒評估 | 社會偏見與公平性 | 多選問答需保持中立回答 | 整合多類對抗提示，測試集規模未公開（數百至千級） [arXiv](https://arxiv.org/abs/2503.19540) | 12 種偏見類型；對抗性提示設計；能揭露傳統評估低估之隱藏風險 | 模型在極端提示下的中立選擇率 |
| **AILuminate** | 聊天系統之安全風險評估 | AI 風險與可靠性 | 多危害類別安全應對 | 每語言 24,000+ 公開練習題、12,000 私有測試題 [MLCommons](https://mlcommons.org/ailuminate/?utm_source=chatgpt.com)[新聞稿發佈與投資者關係服務](https://www.businesswire.com/news/home/20241204285618/en/MLCommons-Launches-AILuminate-First-of-Its-Kind-Benchmark-to-Measure-the-Safety-of-Large-Language-Models?utm_source=chatgpt.com) | 12 大危害分類（暴力、仇恨、隱私…）；公開／私有分割；調校後評估模型集成方法；五級等級制與熵基評分 | 危害類別訂正率與總體安全等級 |
| **MILU** | 印度語系多領域理解能力評估 | 多語種 NLP（印度語） | 多選題橫跨 41 科目 | 85,000 題，覆盖 11 種印度語言 × 8 領域 × 42 科目 [論文與代碼](https://paperswithcode.com/dataset/milu?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2411.02538?utm_source=chatgpt.com) | 文化特定考題（地區考試內容）；英語＋印度語雙語；表格式題型；公開 lm-eval-harness 支援 | 各語言與各領域平均準確率 |
| **GridAgent**<br/>(自演化格網遊戲) | 多模態格網遊戲任務能力評估 | 多模態推理 | 格網遊戲中執行、感知、推理、記憶、學習、規劃 | 12 道獨特格網遊戲題目 [OpenReview](https://openreview.net/forum?id=jpypMKAsO6&utm_source=chatgpt.com) | 多種提示格式（列表、示意、圖像）；避開訓練集中題；自演化任務生成；兼容 LLM-MoE 多模態評估 | 核心能力別得分與任務成功率 |
| **SuperGLUE** | 高難度通用語言理解 | 自然語言理解 | 閉式問答、多選、推理、同指代等 | 8 項任務，合計近 85,000 題（各任務數百到萬級） [arXiv](https://arxiv.org/abs/1905.00537?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/datasets/aps/super_glue?utm_source=chatgpt.com) | 強調低資料量學習；多樣化題型（BoolQ、COPA、WSC、MultiRC…）；公開領先榜 | 各任務平均準確率／F1 分數、綜合平均分 |
| **MultiMedQA** | 醫療領域問答能力 | 醫療臨床 QA | 專業考試題、研究問答、民眾問答 | 結合 6 個公開資料集（MedQA 12.7k、MedMCQA 193k、PubMedQA 1k…）+ 3,173 題 HealthSearchQA；總 ~210k 題 [Nature](https://www.nature.com/articles/s41586-023-06291-2?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/collections/openlifescienceai/multimedqa-66098a5b280539974cefe485?utm_source=chatgpt.com) | 跨領域 QA；多軸人類評估（事實性、理解、風險、偏見）；MCQA + 自由文本 | MCQA 準確率＋人類評分軸平均分 |
| **FinBen** | 財務 NLP 全面評估 | 金融與經濟 | IE、文本分析、問答、生成、風險、預測、決策 | 36（或 42）資料集× 24 項任務，涵蓋 7（或 8）大類 [arXiv](https://arxiv.org/html/2402.12659v2?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2402.12659?utm_source=chatgpt.com) | 首次納入股票交易評估；代理人 + RAG 評估；雙語 (英／西)；開放程式庫与即時榜單 | 各項任務之準確率／F1／盈利指標等綜合指標 |
| **CMMLU** | 中文多任務知識推理 | 中文學科理解 | 多選題覆蓋 67 學科 | 1,528 題，涵蓋 67 個中小學至專業學科 [arXiv](https://arxiv.org/abs/2306.09212?utm_source=chatgpt.com)[GitHub](https://github.com/haonan-li/CMMLU/blob/master/README_EN.md?utm_source=chatgpt.com) | 包含中國特定科目；原創考題來源：模擬考、競賽題；85% 以上題目 OCR 自 PDF 提取，減少模型泄漏 | 各科目平均準確率、相對隨機基線 |
| **LV-Eval** | 長文本單／多跳 QA 能力 | 長上下文理解 | 單跳 QA、Multi 跳 QA | 11 個雙語資料集，5 種長度等級 (16k–256k 字)；平均 102k 字 [arXiv](https://arxiv.org/abs/2402.05136?utm_source=chatgpt.com)[GitHub](https://github.com/infinigence/LVEval?utm_source=chatgpt.com) | 插入混淆事實；關鍵詞替換；關鍵詞召回指標；同 QA 不同文脈長度可控評估 | 多長度場景下得分降幅、關鍵詞召回率 |
| **MT Bench** | 多輪對話與指令遵循 | 聊天／對話系統 | 80 道多輪高質量問題；專家人類配對評分 | 80 個問題 × 3.3k 人類偏好配對 [論文與代碼](https://paperswithcode.com/dataset/mt-bench?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2306.05685v4?utm_source=chatgpt.com) | 8 大提示類別；GPT-4 作為評審之 LLM-as-Judge；Elo 制排行榜；配對比較機制 | 專家配對偏好率 (Elo 分數)、群眾投票 |