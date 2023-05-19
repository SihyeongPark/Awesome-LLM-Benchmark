<p align="center">
  <img src="resources/logo.png" alt="Logo">
</p>

# Awesome-LLM-Benchmark
Awesome-LLM-Benchmark: List of Datasets/benchmarks for Large-Language Models


## Training and Evaluation Datasets/Benchmarks of LLMs
| Model | Paper | Publishing  | Year  | Training  | Evaluation  |
| --- | --- | --- | --- | --- | --- |
| Phrase Representation `RNN` | [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)  | EMNLP | 2014  | newstest2012/2013/2014, MERT  | BLEU  |
| Seq2Seq `LSTM`  | [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) | NeurIPS | 2014  | WMT-14  | BLEU  |
| Transformer `Transformer` | [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)  | NeurIPS | 2017  | WMT-14  | BLEU , PPL  |
| ELMO `LSTM` | [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)  | NAACL | 2018  | SNLI, CoNLL 2012/2002, SQuAD, SST, 1BW  | Coref, NER, SNLI, SQuAD, SRL, SST-5 |
| GPT-1 `Transformer (Decoder)` `GPT` | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)  | Web | 2018  | BookCorpus, 1BW | CoLA, GLUE, MNLI, MRPC, QNLI, QQP, RACE, RTE, SciTail, SNLI, SST-2, STS-B, StoryCloze |
| BERT `Transformer (Encoder) ` `BERT`  | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)  | NAACL | 2018  | GLUE, SQuAD, SWAG | CoLA, CoNLL-2003, GLUE, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B, SWAG, SQuAD |
| MT-DNN `BERT` | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)  | ACL | 2019  | GLUE, SNLI, SciTail | AX, CoLA, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SciTail, SST-2, STS-B, WNLI |
| GPT-2 `GPT` | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Web | 2019  | WebText | 1BW, BLEU, CoQA, PTB, WikiText-2, Wikitext-103, enwik8, text8 |
| MASS `BERT` | [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/pdf/1905.02450.pdf)  | ICML  | 2019  | WMT-14/16 | BLEU, PPL, ROUGE  |
| UniLM `BERT`  | [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf)  | NeurIPS | 2019  | CNN/DailyMail dataset | AX, CoQA, Div, DSTC7, Entropy, GLUE, MCC, METEOR, MNLI, MRPC, NIST, NLG, QNLI, QQP, ROUGE, RTE, SST-2, STS-B, SQuAD, WNLI |
| XLNet `BERT`  | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) | NeurIPS | 2019  | BookCorupus, Wikipedia, Giga5, ClueWeb 2012-B, Common Crwal, SentencePiece  | AG, Amazon-2/5, CoLA, DBpedia, IMDb, MNLI, MRPC, QNLI, QQP, RACE, RTE, SST-2, STS-B, SQuAD, Yelp-2/5  |
| RoBERTa `BERT`  | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) | ArXiv | 2019  | BookCorpus, Wikipidia, CC-News, OpenWebText, Stories  | bsz, CoLA, data, GLUE, lr, MNLI, MRPC, PPL, QNLI, QQP, RACE, RTE, SQuAD, SST-2, steps, STS, WNLI  |
| ALBERT `BERT` | [ALBERT: A lite bert for self-supervised learning of language representations](https://arxiv.org/pdf/1909.11942.pdf)  | ICLR  | 2019  | BookCorpus, Wikipedia | CoLA, GLUE, MNLI, MRPC, QNLI, QQP, RACE, RTE, SQuAD, SST, SST-2, STS, WNLI  |
| Megatron-LM `BERT`  | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf) | ArXiv | 2019  | Wikipedia, CC-Stories, RealNews, OpenWebtext, BookCorpus, LAMBADA | LAMBADA, MNLI, QQP, RACE, SQuAD |
| BART Transformer  | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) | ACL | 2019  | BERT  | BLEU, CNN/DM, CoLA, ConvAI2, ELI5, MNLI, MRPC, QNLI, QQP, RTE, SQuAD, SST, STS-B, WMT-16, XSum  |
| T5 Transformer  | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/volume21/20-074/20-074.pdf) | JMLR  | 2019  | C4, Unfiltered C4, RealNews-like, WebTest-like, Wikipedia, Books CorpusTensorFlow Datasets  | BoolQ, CB, CNN/DM, CoLA, COPA, GLUE, MNLI, MRPC, MultiRC, QNLI, QQP, ReCoRD, RTE, SQuAD, SST-2, STS-B, SuperGLUE, WiC, WNT, WSC |
| DistilBERT `BERT` | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf) | ArXiv | 2019  | Megatron-LM | CoLA, IMDb, MNLI, MRPC, QNLI, QQP, RTE, SQuAD, SST-2, STS-B, WNLI |
| ELETRA `Transformer`  | [ELETRA: Pre-training text encoders as discriminators rather than generators](https://openreview.net/pdf?id=r1xMH1BtvB) | ICLR  | 2020  | BERT, XLNet | CoLA, GLUE, MNLI, NROC, QNLI, QQP, RTE, SQuAD, SST, STS |
| GPT-3 `GPT` | [Language Models are Few-Shot Learners](https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)  | NeurIPS | 2020  | Common Crawl, WebText2, Book21, Books2, Wikipedia, in-context | ARC, Arithmetic, BoolQ, BLEU, CB, CoQA, DROP, HellaSwag, LAMBADA, MultiRC, NaturalQS, OpenBookQA, PIQA, QuAC, RACE, ReCoRD, RTE, StoryCloze, SQuAD, SuperGLUE, TriviaQA, WebQA, Word Unscrambling/Manipulation  |
| DeBERTa `BERT`  | [DeBERTa: Decoding-enhanced bert with disentangled attention](https://openreview.net/pdf?id=XPZIaotutsD)  | ICLR  | 2020  | BERT, Wikipedia, BookCorpus, OPENWEBTEXT, STORIES | BoolQ, CB, CoLA, COPA, GLUE, MNLI, MRPC, MultiRC, NER, QNLI, QQP, RACE, ReCoRD, RTE, SST-2, STS-B, SuperGELU, SQuAD, SWAG, WSC, WiC |
| BigBird `Transformer` | [Big Bird: Transformers for Longer Sequences](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)  | NeurIPS | 2020  | RoBERTa, Books, CC-News, Stories, Wikipedia | Arxiv, BigPatent, CoLA, GLUE, Hyperparisan, IMDb, MNLI, MRPC, NatualQ, PubMed, QNLI, QQP, RTE, SST-2, STS-B, TriviaQA, WikiHop, Yelp-5  |
| Switch Transformer `Transformer - T5` | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)  | JMLR  | 2021  | C4, mC4 | ANLI, ARC, CB Natual QA, CB Trivia QA, CB Web QA, CNN/DM, FLOPs/Seq, GLUE, SQuAD, SuperGLUE, Winogrande, XSum |
| Gopher `Transformer`  | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)  | ArXiv | 2021  | MassiveWeb, Books, C4, News, GitHub, Wikipedia  | BIG-Bench, FEVER, RACE, ... (152 tasks) |
| LaMDA `Transformer` `GPT` | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)  | ArXiv | 2022  | MTB dataset | Datasets for Quality/Safety/Groundedness  |
| InstructGPT `GPT` | [Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) | NeurIPS | 2022  | Prompt (Generation, Open QA, Brainstorming, Chat, Rewrite, Summarization, Classification, Other, Closed QA, Extract), SFT, RM, PPO  | PPO, RM, SFT  |
| MT-NLG `Transformer`  | [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf) | ArXiv | 2022  | Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, Gutenberg (PG-19), BookCorpus2, NIH ExPorter, ArXiv, GitHub, Pile-CC, CC-2020-50, CC-2021-04, RealNews, CC-Stories | BoolQ, HANS, HellaSwag, PIQA, RACE, LAMBADA, WiC, Winogrande  |
| Chinchilla `Transformer`  | [Training Compute-Optimal Large Language Models](https://openreview.net/pdf?id=iBBcRUlOAPR) | NeurIPS | 2022  | MassiveText (Gopher), SentencePiece | Gopher (BIG-Bench, FEVER, RACE, ... ) |
| PaLM `Transformer - Decoder`  | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf) | ArXiv | 2022  | LaMDA/GLam (Webpages, books, Wikipedia, news articles, source code (GitHub), social media conversations)  | ANLI, ARC, BLEU, BoolQ, CB, Clean E2E NLG, COPA, CoQA, Czech Restaurant response generation, DeepFix, DROP, GSM8K-Python, HellaSwag, HumanEval, Hymanities, LAMBADA, MBPP, MMLU, MLSum, MultiRC, Natural Questions, OpenBookQA, PIQA, QuAC, RACE, ReCoRD, ROUGE-2, RTE, Social Sciences, SQuAD, STEM, StoryCloze, SuperGLUE, TransCoder, TriviaQA, TyDiQA-GoldP, WebNLG 2020, WebQA, WiC, WinoGrande, Winograd, WikiLingua, WMT, WSC, XSum  |
| ChatGPT `GPT` | - | - | 2022  | InstructGPT | InstructGPT |
| LLaMA `GPT` | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)  | ArXiv | 2023  | CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, Stack Exchange  | ARC, BoolQ, CrowS-Pairs, GSM8K, HellaSwag, HumanEval, Humanities, MATH, MBPP, MMLU, NaturalQuestions, OBQA, PIQA, RACE, RealToxicityPrompts, SIQA, Social Sciences, STEM, TriviaQA, TruthfulQA, WinoGender, WinoGrande  |
| LLaMA-GPT4 `GPT`  | [INSTRUCTION TUNING WITH GPT-4](https://arxiv.org/pdf/2304.03277.pdf) | ArXiv | 2023  | Alpaca (Elgish Instruction-Following Data), English GPT-4 answer, GPT-4 self-instruct, Chinese Instruction-Following Data | Chinese translation, ROUGE, User-Oriented-Instruction-252, Vinuca-Instructions-80, unnatural Instructions (text-davinci-002)  |
| PaLM 2 `Transformer - Decoder`  | [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf) | Technical Report  | 2023  | PaLM, non-English data  | ANLI, ARC, ARCADE, AUC-ROC, BIG-bench, BLEURT, BoolQ, CB, COPA, DROP, FRMT, GSM8K, HellaSwag, HumanEval, LAMBADA, MATH, MBPP, MGSM, MMLU, MQM, MultiRC, Natural Question, OpenBookQA, PIQA, RACE, ReCoRD, ROUGE, RTE, SQuAD, StategyQA, StoryCloze, TriviaQA, TyDiQA, WebQuestion, WikiLingua, Winograd, WinoGrande, WMT21, WSC, WiC, XCOPA, XLSum, XSum  |


## LLM Datasets/Benchmarks by Category

**Sentiment Analysis (Single-Sentence Tasks)**

- SST-2 (Stanford Sentiment Treebank)
- IMDb (Large Movie Review Dataset)

**Linguistic Acceptability (Single-Sentence Tasks)**

- CoLA (Corpus of Linguistic Acceptability)

**Simliarity and Paraphrase Tasks**

- MRPC (Microsoft Research Paraphrase Corpus)
- QQP (Quora Question Pairs)
- STS-B (Semantic Textual Similarity Benchmark)

**Natural Language Inference (NLI)**

- CB (CommitmentBank)
- RTE (Recognizing Textual Entailment)
- HellaSwag
- MNLI (Multi-Genre Natural Language Inference Corpus)
- RTE (Recognizing Textual Entailment)
- WNLI (Winograd NLI)
- Story Cloze
- SNLI (Stanford Natural Language Inference)
- ANLI (Adversarial NLI)
- SciTail
- AX
- SWAG (Situations With Adversarial Generations)

**Word Sense Disambiguation (WSD)**

- WiC (Word-in-Context)

**Coreference Resolution (coref.)**

- WSC (Winograd Schema Challenge)
- Winogrande

**Prediction**

- LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects)

**Reading Comprehension (QA)**

- BoolQ (Boolean Questions)
- COPA (Choice of Plausible Alternatives)
- PIQA (Physical Interaction QA)
- QNLI (Stanford Question Answering Dataset)
- SQuAD (Stanford Question Answering Dataset)
- MultiRC (Multi-Sentence Reading Comprehension)
- ReCoRD (Reading Comprehension with Commonsense Reasoning Dataset)
- RACE (ReAding Comprehension dataset from Examinations)
- TriviaQA
- CoQA (Conversational Question Answering)
- MultiRC (Multi-Sentence Reading Comprehension)
- NaturalQA
- DROP (Discrete Reasoning Over the content of Paragraphs)
- OpenBookQA
- MBPP (Mostly Basic Programming Problems)
- QuAC (Question Answering in Context)
- Humanities
- Social Science
- STEM (Science, Technology, Engineering, and Mathmatics)

**IQ Test**

- ARC (AI2's Reasoning Challenge)

**Summarization**

- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- XSum (Extreme Summarization)
- CNN/DM
- WikiLingua
- XL-Sum

**Functional Correctness for Sunthesizing**

- HumanEval

**Information Extraction**

- NER (Named Entity Recognition)
- FEVER (Fact Extraction and VERification)

**Math**

- GSM8K (Grade School Math 8.5K)
- MATH (Mathematical problem-solving ability)

## LLM Datasets/Benchmarks List
| Benchmark Suite | Year  | Paper | Publishing  | Code  | Benchmark   |
| --- | --- | --- | --- | --- | --- |
| CoNLL-2003  | 2003  | [Introduction to the CoNLL-2003 Shared Task:Language-Independent Named Entity Recognition](https://aclanthology.org/W03-0419.pdf) | NAACL |   |   |
| ROUGE | 2004  | [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf) | WS  | [GitHub](https://github.com/google-research/google-research/tree/master/rouge)  | ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S Summarization  |
| IMDb  | 2011  | [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015.pdf) | ACL | [Web](http://ai.stanford.edu/~amaas/data/sentiment/)  | Semantic and sentiment similarities among words |
| SST | 2013  | [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170.pdf)  | EMNLP | [Web](https://nlp.stanford.edu/sentiment/)  |   |
| SNLI  | 2015  | [A large annotated corpus for learning natural language inference](https://aclanthology.org/D15-1075.pdf) | EMNLP |   | Classification  |
| Story Cloze | 2016  | [A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://aclanthology.org/N16-1098.pdf)  | NAACL | [Web](https://cs.rochester.edu/nlp/rocstories/) | Generic story understanding evaluation (Not available)  |
| LAMBADA | 2016  | [The LAMBADA dataset:Word prediction requiring a broad discourse context](https://aclanthology.org/P16-1144.pdf)  | ACL | [Web](https://zenodo.org/record/2630551#.YFJVaWT7S_w) | Word prediction task  |
| SQuAD | 2016/2018 | [SQuAD: 100,000+ Questions for Machine Comprehension of Tex](https://aclanthology.org/D16-1264.pdf) | EMNLP/ACL | [Web](https://rajpurkar.github.io/SQuAD-explorer/)  | Question and Answering  |
| TriviaQA  | 2017  | [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://aclanthology.org/P17-1147.pdf) | ACL | [GitHub](https://github.com/mandarjoshi90/triviaqa) | Reading comprehension question and answering  |
| CNN/DM  | 2017  | [Get To The Point: Summarization with Pointer-Generator Networks](https://aclanthology.org/P17-1099.pdf)  | ACL | [GitHub](https://github.com/abisee/cnn-dailymail) | Summarization |
| RACE  | 2017  | [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://aclanthology.org/D17-1082.pdf)  | EMNLP | [GitHub](https://github.com/qizhex/RACE_AR_baselines) | RACE-M, RACE-H, RACE (examinations - word matching, paraphrasing, …)  |
| SRL | 2017  | [Deep Semantic Role Labeling: What Works and What’s Next](https://aclanthology.org/P17-1044.pdf)  | ACL | [GitHub](https://github.com/luheng/deep_s기) | CoNLL-2005/2012 |
| SWAG  | 2018  | [SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](http://aclanthology.lst.uni-saarland.de/D18-1009.pdf)  | EMNLP | [Web](https://rowanzellers.com/swag/) | Grounded Commonsense  |
| FEVER | 2018  | [FEVER: a large-scale dataset for Fact Extraction and VERification](https://aclanthology.org/N18-1074.pdf)  | NAACL | [Web](https://fever.ai/dataset/fever.html)  | Extract |
| OpenBookQA  | 2018  | [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://aclanthology.org/D18-1260.pdf)  | EMNLP | [GitHub](https://github.com/allenai/OpenBookQA) | Questioner and an answerer  |
| QuAC  | 2018  | [QuAC : Question Answering in Context](https://aclanthology.org/D18-1241.pdf) | EMNLP | [Web](https://quac.ai/) | Information seeking dialog  |
| SciTail | 2018  | [SCITAIL: A Textual Entailment Dataset from Science Question Answering](http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf) | AAAI  | [GitHub](https://github.com/allenai/scitail)  | Questioner and an answerer  |
| XSum  | 2018  | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206.pdf) | EMNLP | [GitHub](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) | BBC articles Summarization  |
| DROP  | 2019  | [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246.pdf) | NACCL | [GitHub](https://github.com/allenai/allennlp-models#tasks-and-components) | Reading Comprehension (AllenNLP)  |
| GLUE  | 2019  | [GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7) | ICLR  | [GitHub](https://github.com/nyu-mll/GLUE-baselines) | CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI  |
| SuperGLUE | 2019  | [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://w4ngatang.github.io/static/papers/superglue.pdf) | NeurIPS | [GitHub](https://github.com/nyu-mll/jiant)  | BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC |
| ARC | 2019  | [On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547.pdf)  | ArXiV | [GitHub](https://github.com/fchollet/ARC/)  | Psychometric intelligence tests |
| HellaSwag | 2019  | [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472.pdf)  | ACL | [GitHub](https://github.com/rowanz/hellaswag) | Commonsense question  |
| CoQA  | 2019  | [CoQA: A Conversational Question Answering Challenge](https://aclanthology.org/Q19-1016.pdf)  | TACL  | [Web](https://stanfordnlp.github.io/coqa/)  | Questioner and an answerer  |
| NaturalQS | 2019  | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026.pdf) | TACL  | [Web](https://ai.google.com/research/NaturalQuestions/dataset)  | Questioner and an answerer  |
| ANLI  | 2020  | [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://aclanthology.org/2020.acl-main.441.pdf) | ACL | [GitHub](https://github.com/facebookresearch/anli)  | A1, A2, A3, S |
| PIQA  | 2020  | [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/pdf/1911.11641.pdf)  | AAAI  | [Web](https://yonatanbisk.com/piqa/)  | Knowledge of physical commonsense |
| WinoGrande  | 2020  | [WINOGRANDE: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/pdf/1907.10641.pdf) | AAAI  | [GitHub](https://github.com/allenai/winogrande) | WSC |
| WiKiLingua  | 2020  | [WikiLingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization](https://aclanthology.org/2020.findings-emnlp.360.pdf) | EMNLP | [GitHub](https://github.com/esdurmus/Wikilingua)  | Cross-Lingual Abstractive |
| HumanEval | 2021  | [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)  | ArXiv | [GItHub](https://github.com/openai/human-eval)  | Hand-Written Evaluation Set |
| MATH  | 2021  | [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/pdf/2103.03874.pdf)  | NeurIPS | [GitHub](https://github.com/hendrycks/math/)  | Competition mathematics problems  |
| MBPP  | 2021  | [Program Synthesis with Large Language Models](https://arxiv.org/pdf/2108.07732.pdf)  | CoRR  | [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) | Python Code |
| MULL  | 2021  | [Measuring massive multitask language understanding](https://openreview.net/pdf?id=d7KBjmI3GmQ) | ICLR  | [GitHub](https://github.com/hendrycks/test) | Massive multitask test consisting of multiple-choice questions  |
| GSM8K | 2021  | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168.pdf)  | ArXiv | [GitHub](https://github.com/openai/grade-school-math) | Grade school math problems  |
| BIG-bench | 2022  | [Beyond the imitation game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/pdf/2206.04615.pdf)  | ArXiv | [GitHub](https://github.com/google/BIG-bench) | 214 Tasks Benchmark |
| XLSum | 2021  | [XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages](https://aclanthology.org/2021.findings-acl.413.pdf) | ACL | [GitHub](https://github.com/csebuetnlp/xl-sum)  | 1 million professionally annotated article-summary pairs from BBC |

## Support for LLM Datasets/Benchmarks in Deep Learning Framewors
