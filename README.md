# CLLE
Code and Datasets of " CLLE: A Benchmark for Continual Language Learning Evaluation in Multilingual Machine Translation".

Accepted by findings of EMNLP2022.

### Abstract
Continual Language Learning (CLL) in multilingual  translation is inevitable when new languages are required to be translated. Due to the lack of unified and generalized benchmarks, the evaluation of existing methods is greatly influenced by experimental design which usually has a big gap from the industrial demands. In this work, we propose the first ContinualLanguage Learning Evaluation benchmark CLLE in multilingual translation. CLLE consists of a Chinese-centric corpus --- CN-25 and two CLL tasks --- the close-distance language continual learning task and the language family continual learning task designed for real and disparate demands. Different from existing translation benchmarks, CLLE considers several restrictions for CLL, including domain distribution alignment, content overlap, language diversity, and the balance of corpus. Furthermore, we propose a novel framework COMETA based on constrained optimization and meta-learning to alleviate catastrophic forgetting and dependency on historical training data by using a meta-model to retain the important parameters for old languages. Our experiments prove that CLLE is a challenging CLL benchmark and that our proposed method is effective when compared with other strong baselines. Due to the construction of corpus, the task designing and the evaluation method are independent of the central language, we also construct and release the English-centric corpus EN-25 to facilitate academic research.

### EN-25
  https://drive.google.com/drive/folders/1Vm9A_SnjVvOiEhCj30azYeEZFmUi5i1g?usp=sharing
### CN-25 
  https://drive.google.com/drive/folders/1zlHS9vGBmxJe-NxM24UZxmJnI9qf-zko?usp=sharing
### COMETA
  branch : master
