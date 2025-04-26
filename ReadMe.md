事實性（Factuality）評測數據集

| 資料集 | 測試重點 | 主要領域 | 核心任務類型 | 資料規模 | 獨特功能／技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **BoolQ** | 測試模型對自然產生的是/否問題的理解與推理難度 [ACL Anthology](https://aclanthology.org/N19-1300/?utm_source=chatgpt.com) | 閱讀理解／問答 | Yes/No 問答 | 15,942 範例 [GitHub](https://github.com/google-research-datasets/boolean-questions?utm_source=chatgpt.com) | 問題源自真實搜尋查詢，出題無引導，難度高 [paperswithcode.com](https://paperswithcode.com/dataset/boolq?utm_source=chatgpt.com) |
| **NaturalQuestions-Closed & Retrieved** | 測試模型在長／短答案設定下的開放域問答能力 [ACL Anthology](https://aclanthology.org/Q19-1026.pdf?utm_source=chatgpt.com) | 開放域問答 | 長答案選擇 & 短答案抽取 | 307,373 訓練 / 7,830 驗證 / 7,842 測試 [paperswithcode.com](https://paperswithcode.com/dataset/natural-questions?utm_source=chatgpt.com) | 同時標註長段落與短實體答案，真實用戶查詢來源 [ACL Anthology](https://aclanthology.org/Q19-1026.pdf?utm_source=chatgpt.com) |
| **RealTimeQA** | 測試對「當前世界」動態事實的即時問答能力 [arXiv](https://arxiv.org/abs/2207.13332?utm_source=chatgpt.com) | 動態 QA | 時間敏感問題回答 | 持續更新；示例數量隨時間推移變化 [arXiv](https://arxiv.org/abs/2207.13332?utm_source=chatgpt.com) | 週期性發布最新問題，強調時效性與檢索 [arXiv](https://arxiv.org/html/2207.13332v2?utm_source=chatgpt.com) |
| **TyDiQA-noContext & TyDiQA-goldP** | 測試多語言信息檢索式問答（無上下文 vs. 金段） [Hugging Face](https://huggingface.co/datasets/khalidalt/tydiqa-goldp?utm_source=chatgpt.com)[paperswithcode.com](https://paperswithcode.com/dataset/tydiqa-goldp?utm_source=chatgpt.com) | 多語言問答 | 段落檢索 + 答案 Span 抽取 | 原始問答對約 204K；GoldP 訓練 50 MB/驗證 10 MB [GitHub](https://github.com/google-research-datasets/tydiqa?utm_source=chatgpt.com) | 涵蓋 11 種語言，問題非翻譯生成，避免跨語偏差 [ACL Anthology](https://aclanthology.org/2020.tacl-1.30.pdf?utm_source=chatgpt.com) |
| **CSQA** | 測試知識圖譜上串聯 QA 對話中複雜推理能力 [arXiv](https://arxiv.org/abs/1801.10314?utm_source=chatgpt.com)[SpringerLink](https://link.springer.com/article/10.1007/s10115-022-01744-y?utm_source=chatgpt.com) | 對話式 KG 問答 | 多步邏輯／量化推理 + 上下文核心詞消解 | 約 200K 對話，1.6M 回合；12.8M 實體 [ACL Anthology](https://aclanthology.org/2021.eacl-main.72.pdf?utm_source=chatgpt.com) | 結合對話與 KG 多元推理，需檢索子圖回答 [arXiv](https://arxiv.org/abs/1801.10314?utm_source=chatgpt.com) |
| **MT Bench** | 多輪對話與指令遵循 | 聊天／對話系統 | 80 道多輪高質量問題；專家人類配對評分 | 80 個問題 × 3.3k 人類偏好配對 [論文與代碼](https://paperswithcode.com/dataset/mt-bench?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2306.05685v4?utm_source=chatgpt.com) | 8 大提示類別；GPT-4 作為評審之 LLM-as-Judge；Elo 制排行榜；配對比較機制 |
| **TriviaQA** | 閱讀理解與證據檢索結合之開放領域問答 | 閱讀理解＋檢索 QA | 開放領域問答（證據三元組） | 650,000 題 + 平均 6 篇證據段 | Trivia 愛好者提問；Bing/Wiki 多文件證據支援 |
| **CommonsenseQA** | 測試機器對日常常識知識的運用能力 | 常識問答 | 5 選 1 多選題 | 12,247 題 [論文與代碼](https://paperswithcode.com/dataset/commonsenseqa?utm_source=chatgpt.com) | 基於 ConceptNet 自動生成問題與干擾選項 [ACL Anthology](https://aclanthology.org/N19-1421.pdf?utm_source=chatgpt.com) |
| **AGIEval** | 評估基礎模型在真實人類標準化考試（SAT、高考、LSAT、律師資格、公務員考試）中的通用能力 [ACL Anthology](https://aclanthology.org/2024.findings-naacl.149/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2304.06364?utm_source=chatgpt.com) | 通用認知能力 | 多科目多選題 | 取材自 20 項公開高標準考試試題 [GitHub](https://github.com/ruixiangcui/AGIEval?utm_source=chatgpt.com) | 使用真實人類考試試題；雙語設計；涵蓋廣泛領域；無人工構造數據 [ACL Anthology](https://aclanthology.org/2024.findings-naacl.149/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2304.06364?utm_source=chatgpt.com) |
| **TruthfulQA (MC2)** | 評估模型在面對常見謬誤資訊時的真實回答能力 | 事實性 QA | 多選題 | 817 題 | 涵蓋健康、法律、金融、政治領域之常見誤導；多選設計抵抗假訊息 |
|  |  |  |  |  |  |
| **Social IQA** | 測試社交情境中情感與行為推理能力 | 社交常識 | 3 選 1 多選題 | 38,000 題 [maartensap.com](https://maartensap.com/social-iqa/?utm_source=chatgpt.com)[ACL Anthology](https://aclanthology.org/D19-1454/?utm_source=chatgpt.com) | 提供多樣社交場景； mitigates 誤答樣式偏差的新框架 [ACL Anthology](https://aclanthology.org/D19-1454/?utm_source=chatgpt.com) |

**長上下文（Long Context）評測數據集**

| **NarrativeQA** | 測試對長篇故事（書籍／電影稿）的閱讀理解與敘事推理 [ACL Anthology](https://aclanthology.org/Q18-1023/?utm_source=chatgpt.com) | 長文本閱讀理解 | 自由格式 QA | 1,572 本書／劇本，約 45K QA 對 [MIT Press Direct](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00411/107386/Narrative-Question-Answering-with-Cutting-Edge?utm_source=chatgpt.com)[NLP-progress](https://nlpprogress.com/english/question_answering.html?utm_source=chatgpt.com) | 同時提供摘要與全文模式，強化綜合推理 [MIT Press Direct](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00411/107386/Narrative-Question-Answering-with-Cutting-Edge?utm_source=chatgpt.com)[ACL Anthology](https://aclanthology.org/Q18-1023/?utm_source=chatgpt.com) |
| --- | --- | --- | --- | --- | --- |
| **SCROLLS-Qasper & SCROLLS-Quality** | 測試長上下文下多任務性能：QA、NLI、摘要等 [GitHub](https://github.com/tau-nlp/scrolls?utm_source=chatgpt.com)[paperswithcode.com](https://paperswithcode.com/dataset/scrolls?utm_source=chatgpt.com) | 長文本綜合評測 | 抽取式/生成式 QA 及 NLI | QASPER: train 2,567/dev 1,726/test 1,399；Quality: 2,523/2,086/2,128 [TensorFlow](https://www.tensorflow.org/datasets/catalog/scrolls?utm_source=chatgpt.com) | 統一多任務標準化長文本基準，涵蓋多種任務 [ACL Anthology](https://aclanthology.org/2022.emnlp-main.823.pdf?utm_source=chatgpt.com) |
| **XL-Sum** | 測試多語言新聞抽象式摘要能力 [ACL Anthology](https://aclanthology.org/2021.findings-acl.413/?utm_source=chatgpt.com) | 多語言摘要生成 | 多語種文章→摘要生成 | 約 1M 對，涵蓋 44 種語言 [arXiv](https://arxiv.org/abs/2106.13822?utm_source=chatgpt.com) | BBC 專業標注，跨 44 語言高質量抽象摘要 [paperswithcode.com](https://paperswithcode.com/dataset/xl-sum?utm_source=chatgpt.com) |
| **LV-Eval** | 長文本單／多跳 QA 能力 | 長上下文理解 | 單跳 QA、Multi 跳 QA | 11 個雙語資料集，5 種長度等級 (16k–256k 字)；平均 102k 字 [arXiv](https://arxiv.org/abs/2402.05136?utm_source=chatgpt.com)[GitHub](https://github.com/infinigence/LVEval?utm_source=chatgpt.com) | 插入混淆事實；關鍵詞替換；關鍵詞召回指標；同 QA 不同文脈長度可控評估 |

專業科目

| **MultiMedQA** | 醫療領域問答能力 | 醫療臨床 QA | 專業考試題、研究問答、民眾問答 | 結合 6 個公開資料集（MedQA 12.7k、MedMCQA 193k、PubMedQA 1k…）+ 3,173 題 HealthSearchQA；總 ~210k 題 [Nature](https://www.nature.com/articles/s41586-023-06291-2?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/collections/openlifescienceai/multimedqa-66098a5b280539974cefe485?utm_source=chatgpt.com) | 跨領域 QA；多軸人類評估（事實性、理解、風險、偏見）；MCQA + 自由文本 | MCQA 準確率＋人類評分軸平均分 |
| --- | --- | --- | --- | --- | --- | --- |
| **FinBen** | 財務 NLP 全面評估 | 金融與經濟 | IE、文本分析、問答、生成、風險、預測、決策 | 36（或 42）資料集× 24 項任務，涵蓋 7（或 8）大類 [arXiv](https://arxiv.org/html/2402.12659v2?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2402.12659?utm_source=chatgpt.com) | 首次納入股票交易評估；代理人 + RAG 評估；雙語 (英／西)；開放程式庫与即時榜單 | 各項任務之準確率／F1／盈利指標等綜合指標 |
| **LiveBench** | 蛋白質結構預測之迭代評價 | 生物信息學 | 同源建模與二級結構預測 | 每週新釋出之數百至千餘蛋白靶標；累計上千場測試 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2373940/?utm_source=chatgpt.com)[Wikipedia](https://en.wikipedia.org/wiki/LiveBench?utm_source=chatgpt.com) | 持續自動化評估；即時比對新入庫 PDB 結構；多指標（RMSD、MAXSUB、對齊準確率）綜合排名 | 以結構重疊分數（RMSD、MAXSUB）與對齊精度排名 |
| **Humanity’s Last Exam (HLE)** | 多模態前沿學術題庫答題準確度評估 | 學術知識綜合 | 閉式多選與填空題回答 | 2,500 道題，涵蓋 100+ 學科，多模態題型 10% 以上 [AGI Safe](https://agi.safe.ai/) | 私有測試集防止資料洩漏；題庫難度動態更新；結合圖文本題目 | 準確率及跨學科平均分比較 |
| **GPQA** | 研究生級科學題（物理、化學、生物） | 數學推理 | 多選題 | 448 題 | 由領域專家出題，針對檢索與共現方法均答錯的困難題進行專項評估 [arXiv](https://arxiv.org/abs/2311.12022?utm_source=chatgpt.com) | 正答率與“Diamond”最難子集準確率 |

數學/科學數據

| **GSM8K** | 測試小學數學多步推理解題能力 [arXiv](https://arxiv.org/abs/2110.14168?utm_source=chatgpt.com) | 數學推理 | 文字算數題解答 + 驗證器排序 | 8.5K 題（7.5K 訓練 + 1K 測試） [paperswithcode.com](https://paperswithcode.com/dataset/gsm8k?utm_source=chatgpt.com) | 高質量人工編題；2–8 步驟；引入 verifiers 提升策略 [arXiv](https://arxiv.org/abs/2110.14168?utm_source=chatgpt.com) |
| --- | --- | --- | --- | --- | --- |
| **MATH** | 測試競賽級數學問題解題與步驟生成 [arXiv](https://arxiv.org/abs/2103.03874?utm_source=chatgpt.com) | 數學問題解決 | 文字數學題 + Step-by-Step 解答 | 12,500 題（7.5K train + 5K test） [arXiv](https://arxiv.org/abs/2103.03874?utm_source=chatgpt.com) | 每題帶完整步驟解答；輔助 AMPS 預訓練資料集 [MATH-AI](https://mathai-iclr.github.io/papers/papers/MATHAI_24_paper.pdf?utm_source=chatgpt.com) |
| **MMLU** | 測試模型在 57 學科多選題上的知識廣度與深度 [arXiv](https://arxiv.org/abs/2009.03300?utm_source=chatgpt.com) | 通用知識理解 | 57 科目多選題 | 約 16,000 題 [arXiv](https://arxiv.org/abs/2009.03300?utm_source=chatgpt.com) | 無需微調下檢測預訓練知識；多領域考試題 [arXiv](https://arxiv.org/abs/2009.03300?utm_source=chatgpt.com) |
| **EXEQ-300k & OFEQ-10k** | 測試數學社群問題的標題生成能力 [clgiles.ist.psu.edu](https://clgiles.ist.psu.edu/pubs/AAAI20-QAmath.pdf?utm_source=chatgpt.com) | 文本摘要／標題生成 | 從詳細數學問句生成簡潔標題 | EXEQ-300k: 290,479；OFEQ-10k: 12,548 [paperswithcode.com](https://paperswithcode.com/dataset/exeq-300k?utm_source=chatgpt.com)[paperswithcode.com](https://paperswithcode.com/dataset/ofeq-10k?utm_source=chatgpt.com) | 同步處理文本與 LaTeX 方程；來源真實問答社群 [clgiles.ist.psu.edu](https://clgiles.ist.psu.edu/pubs/AAAI20-QAmath.pdf?utm_source=chatgpt.com) |
| GaoKao-Benchmark | 中國高考及其他入學考試題目 | 教育測試 | 多選題 | ~22,000題 | 跨學科真實考試場景 |
| C-Eval | 官方多學科中文考試題 | 學術評估 | 多選題 | ~52,000題 | 多層次、多學科 |
| **SCALAR** | 長文本科學文獻之引用邏輯推理評估 | 科學文獻理解 | 引用填空（cloze）式推理 | 基於 ICLR 2025 論文及其引用網絡，動態生成數千條測試 [arXiv](https://arxiv.org/abs/2502.13753) | 無需人工標註；難度可控；動態更新機制防止模型「見過即答」 |
| **CMMLU** | 中文多任務知識推理 | 中文學科理解 | 多選題覆蓋 67 學科 | 1,528 題，涵蓋 67 個中小學至專業學科 [arXiv](https://arxiv.org/abs/2306.09212?utm_source=chatgpt.com)[GitHub](https://github.com/haonan-li/CMMLU/blob/master/README_EN.md?utm_source=chatgpt.com) | 包含中國特定科目；原創考題來源：模擬考、競賽題；85% 以上題目 OCR 自 PDF 提取，減少模型泄漏 |
| **ARC Challenge** | 測試 Grade 3–9 科學考試多選題推理能力 | 科學與抽象推理 | 多選題 | 挑戰集 2,590、簡易集 5,197；共 7,787 題 [TensorFlow](https://www.tensorflow.org/datasets/catalog/ai2_arc?utm_source=chatgpt.com)[ar5iv](https://ar5iv.labs.arxiv.org/html/1803.05457?utm_source=chatgpt.com) | 挑戰集選自 IR 與 PMI 方法失敗題；附 14.3M 科學文本庫 [TensorFlow](https://www.tensorflow.org/datasets/catalog/ai2_arc?utm_source=chatgpt.com)[論文與代碼](https://paperswithcode.com/dataset/arc?utm_source=chatgpt.com) |
| **OpenBookQA** | 開卷式小學科學問答 | 科學常識推理 | 多選題 | 5,957 題（4,957 train / 500 dev / 500 test） [論文與代碼](https://paperswithcode.com/dataset/openbookqa?utm_source=chatgpt.com) | 模仿開卷考試；原創 1,326 條科學事實與 5,167 條群眾常識庫 [論文與代碼](https://paperswithcode.com/dataset/openbookqa?utm_source=chatgpt.com) |
| **DROP** | 閱讀理解＋離散計算（計數、排序、加減） | 數學推理 | 問答（QA） | 96K QAs，6.7K 段落 | 對抗式收集，需解析多重參照並做離散運算，測試更綜合段落理解 [arXiv](https://arxiv.org/abs/1903.00161?utm_source=chatgpt.com) |

## 推理 (Reasoning) 評測數據集

| 基準名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能/技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **BBH** | 聚焦最具挑戰性任務，測量模型當前極限 | 高階推理與任務解決 | 23 大類任務（27 子任務） | 約數千題 | 精選先前模型表現不及人類平均分之困難任務 |
| CLRS | 演算法推理 | 算法 | 輸入輸出對＋推理步驟 | 30種算法大量輸入輸出對 | 提供詳細中間步驟 |
| ProofWriter | 自然語言證明生成 | 邏輯推理 | 證明生成 | ~845,000證明三元組 | 多步邏輯證明＋反向解釋 |
| Fermi Problems | 費米估算問題 | 數值推理 | 估算任務 | 250題 | 多步估算過程 |
| LAMBADA | 長距離文本依賴 | 文本理解 | 最後單詞預測 | 10,022篇段落 | 廣義上下文依賴 |
| HellaSwag | 對抗式常識推理 | NLI | 下一句選擇 | 70,000題 | 對抗式篩選機器生成選項 |
| PIQA | 物理常識二選一 | QA | 多選題 | 16,113訓練＋1,838驗證題 | 日常互動場景 |
| **CO₂ Cost** | 模型推理之碳足跡評估 | NLP 推理效率 | 多基準推理輸出之能源消耗衡量 | 涵蓋 2,742 款模型在多項任務上的推理記錄 [Hugging Face](https://huggingface.co/blog/leaderboard-emissions-analysis?utm_source=chatgpt.com) | 統一硬體與並行策略計算 CO₂ 排放量；按能效比（CO₂／回合）呈現口碑排行 |
| **GridAgent**(自演化格網遊戲) | 多模態格網遊戲任務能力評估 | 多模態推理 | 格網遊戲中執行、感知、推理、記憶、學習、規劃 | 12 道獨特格網遊戲題目 [OpenReview](https://openreview.net/forum?id=jpypMKAsO6&utm_source=chatgpt.com) | 多種提示格式（列表、示意、圖像）；避開訓練集中題；自演化任務生成；兼容 LLM-MoE 多模態評估 |
| **ANLI** | 測試對抗式自然語言推理能力 | 自然語言推理 | 句子蘊涵判斷（三分類：Entail/Neutral/Contradiction） | 訓練 162,865 + 驗證/測試約 3,200 範例 [TensorFlow](https://www.tensorflow.org/datasets/catalog/anli?utm_source=chatgpt.com) | 人機迭代收集難例，三輪不同對抗者設計 [Stanford University](https://web.stanford.edu/class/cs224u/2021/slides/cs224u-2021-nli-part2-handout.pdf?utm_source=chatgpt.com)[ACL Anthology](https://aclanthology.org/2022.scil-1.3.pdf?utm_source=chatgpt.com) |
| **WinoGrande** | 評測代詞消解與常識推理 | 代詞解析 | 填空式多選題 | 44,000 題 | 基於 Winograd Schema 擴展規模；防止原始題目飽和 |

## 摘要 (Summarization) 評測數據集

| 基準名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能/技術亮點 |
| --- | --- | --- | --- | --- | --- |
| WikiLingua | 跨語言 How-to 摘要 | 摘要 | 跨語言摘要 | ~770,000對（18語） | 圖像對齊標準 |
| XSum | 單句極限摘要 | 摘要 | 單句新聞摘要 | 226,711篇 BBC 文章 | 即時新聞單句摘要 |

### 語言理解與翻譯相關評測

| 名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能 / 技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **WMT22** | 機器翻譯 | 多語言翻譯 | 翻譯品質評估 | 多語言多方向 | 年度翻譯比賽、人工與自動評估 |
| **WMT23** | 機器翻譯 | 多語言翻譯 | 翻譯品質與LLM比較 | 多語言多方向 | LLM在翻譯領域的首次深入分析 |
| **FRMT** | 少樣本翻譯 | 區域語系 | Few-shot 翻譯 | 多個語區語言 | 支援低資源語言 Few-shot |
| **TydiQA** | QA系統評估 | 多語言 | 資訊檢索型問答 | 11語言 | 多語言問答、Typologically多樣語系 |
| **MGSM** | 多語鏈式推理 | 數學推理 | Chain-of-thought | 多語數學題 | 多語言 + CoT能力評估 |
| **MMLU** | 學術知識 | 廣泛學科 | 多選題理解 | 57科目，約14k題 | 模擬人類考試難度分布 |
| **NTREX-128** | 翻譯 | 多語新聞 | MT品質評估 | 128語言 | 首個大規模新聞參考集 |
| **FLORES-200** | 機器翻譯 | 多語新聞 | 翻譯能力 | 200語言 | 支援極低資源語言、統一格式 |
| **SuperGLUE** | 高難度通用語言理解 | 自然語言理解 | 閉式問答、多選、推理、同指代等 | 8 項任務，合計近 85,000 題（各任務數百到萬級） [arXiv](https://arxiv.org/abs/1905.00537?utm_source=chatgpt.com)[Hugging Face](https://huggingface.co/datasets/aps/super_glue?utm_source=chatgpt.com) | 強調低資料量學習；多樣化題型（BoolQ、COPA、WSC、MultiRC…）；公開領先榜 |
| **MILU** | 印度語系多領域理解能力評估 | 多語種 NLP（印度語） | 多選題橫跨 41 科目 | 85,000 題，覆盖 11 種印度語言 × 8 領域 × 42 科目 [論文與代碼](https://paperswithcode.com/dataset/milu?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2411.02538?utm_source=chatgpt.com) | 文化特定考題（地區考試內容）；英語＋印度語雙語；表格式題型；公開 lm-eval-harness 支援 |
| **MMLU-Pro** | 多領域跨學科知識准確性 | 通用知識 | 多選題 | ~1000 題（專業題庫） | 包含大學入學、法律、醫學等 57 個學科的高標準考試題，專門針對專業深度與廣度評估模型知識能力 [arXiv](https://arxiv.org/abs/2503.10497?utm_source=chatgpt.com) |

---

### 🖼 圖像理解與多模態評測

| 名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能 / 技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **MMMU** | 多學科圖文推理 | 多模態知識 | 跨模態推理 | 約100k問答 | 專業學科+圖像+文字整合 |
| **TextVQA** | 圖中文字問答 | 圖像理解 | 圖文VQA | 45k圖像 | 強調閱讀圖中文字 |
| **DocVQA** | 文件理解 | 文檔圖像 | VQA | 約50k問題 | 文件格式理解與提問 |
| **ChartQA** | 圖表理解 | 統計圖表 | QA + 邏輯 | 多種圖表類型 | 視覺 + 邏輯推理整合 |
| **InfographicVQA** | 資訊圖理解 | 海報、圖解 | QA | 多樣資訊圖 | 多視覺元素分析 |
| **MathVista** | 圖文數學推理 | 數學 + 視覺 | 推理 / QA | 多題型視覺題 | 數學推理 + 視覺融合 |
| **AI2D** | 圖示推理 | 教育/科學 | 圖文理解 | 15k張圖示 | 評估圖示推理能力 |
| **VQAv2** | 視覺問答 | 圖像理解 | QA | 約265k圖問對 | 增強V問題比重 |
| **XM3600** | 多語多模態 | 跨語+圖文 | QA / Caption | 3600樣本，20語 | 多語多模態 |
| **Memecap** | Meme 理解 | 社交媒體 | Caption / 解釋 | 約9k meme圖 | 對 meme 文字 + 圖融合理解 |

### 🎥 影片理解（Video Understanding）評測資料集

| 資料集名稱 | 測試重點 | 主要領域 | 核心任務類型 | 資料規模與技術亮點 |
| --- | --- | --- | --- | --- |
| **VATEX** | 多語言影片描述 | 多模態理解 | 影片字幕生成、翻譯 | 超過 41,250 部影片，包含 825,000 條中英文字幕，支援影片描述與機器翻譯研究。 |
| **YouCook2** | 程序學習與影片分段 | 教學影片 | 程序分段、密集字幕生成 | 約 2,000 部烹飪教學影片，提供手動標註的程序分段與描述，適用於程序學習與事件解析。 |
| **NExT-QA** | 時序推理與因果理解 | 影片問答 | 多選與開放式問答 | 包含多種問答類型，針對因果與時序推理，提升影片理解從描述到解釋的能力。 |
| **ActivityNet-QA** | 複雜影片問答 | 網路影片 | 問答系統 | 包含 58,000 條問答對，涵蓋 5,800 部複雜網路影片，支援影片問答研究。 |
| **Perception Test MCQA** | 多模態推理與診斷 | 多模態影片 | 多選問答、物件追蹤、音訊分段 | 包含 11,600 部影片，平均長度 23 秒，涵蓋記憶、抽象、物理與語意等推理能力，支援零樣本與少樣本評估。 |

---

### 🔊 音訊處理（Audio）評測資料集

| 資料集名稱 | 測試重點 | 主要領域 | 核心任務類型 | 資料規模與技術亮點 |
| --- | --- | --- | --- | --- |
| **FLEURS** | 多語言語音表示學習 | 語音辨識與翻譯 | 語音辨識、語言識別、翻譯 | 涵蓋 102 種語言，每種語言約 12 小時的語音資料，支援多語言語音任務。 |
| **VoxPopuli** | 表示學習與半監督學習 | 多語言語音 | 語音辨識、翻譯 | 提供 400,000 小時未標註語音資料（23 種語言）與 17,300 小時標註資料（15 種語言），支援大規模語音學習。 |
| **LibriSpeech** | 英語語音辨識 | 自然語音 | 自動語音辨識（ASR） | 約 1,000 小時的英語朗讀語音資料，來源於 LibriVox 公共領域有聲書，廣泛用於 ASR 研究。 |
| **CoVoST 2** | 多語言語音翻譯 | 語音翻譯 | 語音到文本翻譯（ST） | 涵蓋 21 種語言到英語，以及英語到 15 種語言的翻譯資料，支援低資源語言對的研究。 |

---

### 💻 程式碼生成（Coding）評測資料集

| 名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能 / 技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **CodeXGLUE** | 程式理解與生成 | 多語言程式碼 | 多任務（如摘要、翻譯、錯誤修復） | 包含 10 個任務、14 個資料集，提供 BERT、GPT 等基準模型，促進程式碼理解與生成研究。 |  |
| **CodeContests** | 競賽級程式碼生成 | 演算法與競賽程式 | 高難度程式碼生成 | 由 AlphaCode 系統開發，針對競賽級問題進行程式碼生成，展現深層推理能力。 |  |
| **APPS** | 程式設計挑戰能力評估 | Python 程式設計 | 程式碼生成與測試 |  |  |
| **DynaCode** | 動態複雜度感知之程式碼生成評估 | 程式碼生成 | LLM 生成嵌套函式程式碼 | 可生成最高 1.89 億 道唯一程式問題，涵蓋 4 種複雜度單元 × 16 種呼叫圖 [arXiv](https://arxiv.org/abs/2503.10452?utm_source=chatgpt.com) | 動態資料集避免抄襲；複雜度與呼叫圖結合度量；可視化回歸性能降幅隨複雜度上升 |
| **MBPP** | 測試 AI 解決入門級程式設計任務能力 | 程式碼生成 | Python 入門程式設計問題 | 約 1,000 題 | 眾包題目；專為初學者設計 |
| **HumanEval** | 評估 LLM 生成程式碼功能正確性 | 程式碼生成 | Python 函數實現 + 單元測試 | 164 題 | 手工設計多元單元測試用例；函數簽名 + 文檔字符串 |
| **HumanEval+** | 進階程式碼生成＋連續推理 | 編程生成 | Self-invoking 代碼生成 | 164 題擴展版 | 延伸為需調用自生成功能的“自觸發”題，考查模型調用自身函數能力 [arXiv](https://arxiv.org/html/2412.21199v1?utm_source=chatgpt.com) |
| **MultiPL-E** | 多語言程式碼生成評估 | 編程生成 | 跨語言單元測試程式生成 | Python 基準題翻譯至 18–22 種語言 | 一鍵將 HumanEval/MBPP 平行轉換到多語言，衡量多語言泛化能力 [arXiv](https://arxiv.org/abs/2208.08227?utm_source=chatgpt.com) |
| **API-Bank** | LLM 的 API 使用與調用規劃 | 工具使用 | 多回合 API 調用規劃與執行 | 314 對話、753 呼叫； 1,888 訓練對話 | 涵蓋 73 類 API 工具，並提供訓練集＋評估系統，測試規劃與執行能力 [arXiv](https://arxiv.org/abs/2304.08244?utm_source=chatgpt.com) |
| **APIBench** | LLM API 推薦與調用 | 工具使用 | 單步 API 調用 | 1,600+ APIs | 源自 Gorilla，包含多個開源／TorchHub／TensorHub API，減少幻覺 [arXiv](https://arxiv.org/abs/2305.15334?utm_source=chatgpt.com) |
| **Nexus Function Calling** | 零樣本／多步函數調用 | 工具使用 | 單步／並行／嵌套函數呼叫 | 9 類任務、840 測試用例 | 評估零樣本下模型生成單步、並行及嵌套調用能力 [Medium](https://medium.com/data-science/ai-agents-the-intersection-of-tool-calling-and-reasoning-in-generative-ai-ff268eece443?utm_source=chatgpt.com) |
| **BFCL** | 通用函數呼叫能力 | 工具使用 | 單步／多步／多回合函數呼叫 | ~2K 問題－函數配對 | 執行可執行代碼、涵蓋多語言、多場景； V3 添加多回合評估 [Gorilla](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html?utm_source=chatgpt.com) |

人工智慧的倫理與可靠性

| 名稱 | 測試重點 | 主要領域 | 核心任務類型 | 數據規模 | 獨特功能 / 技術亮點 |
| --- | --- | --- | --- | --- | --- |
| **FLEX** | 極端對抗性場景下之公平性魯棒評估 | 社會偏見與公平性 | 多選問答需保持中立回答 | 整合多類對抗提示，測試集規模未公開（數百至千級） [arXiv](https://arxiv.org/abs/2503.19540) | 12 種偏見類型；對抗性提示設計；能揭露傳統評估低估之隱藏風險 |
| **AILuminate** | 聊天系統之安全風險評估 | AI 風險與可靠性 | 多危害類別安全應對 | 每語言 24,000+ 公開練習題、12,000 私有測試題 [MLCommons](https://mlcommons.org/ailuminate/?utm_source=chatgpt.com)[新聞稿發佈與投資者關係服務](https://www.businesswire.com/news/home/20241204285618/en/MLCommons-Launches-AILuminate-First-of-Its-Kind-Benchmark-to-Measure-the-Safety-of-Large-Language-Models?utm_source=chatgpt.com) | 12 大危害分類（暴力、仇恨、隱私…）；公開／私有分割；調校後評估模型集成方法；五級等級制與熵基評分 |