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

## Support for LLM Datasets/Benchmarks in Deep Learning Framewors
