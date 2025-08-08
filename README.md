# 🌐 CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting

**CAMEF** is a novel framework for forecasting financial market trends by integrating **time-series data** and **macroeconomic policy texts**, enhanced by **causal learning through LLM-generated counterfactuals**.  
It was accepted at **KDD 2025**.

- 📄 [arXiv Paper](https://arxiv.org/abs/2502.04592)
- 📊 [Poster (PPT)](https://github.com/lakebodhi/CAMEF/blob/main/resources/CAMEF_KDD2025_Poster.pdf)
- 📚 Citation (BibTeX):
  ```bibtex
  @inproceedings{10.1145/3711896.3736872,
author = {Zhang, Yang and Yang, Wenbo and Wang, Jun and Ma, Qiang and Xiong, Jie},
title = {CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3736872},
doi = {10.1145/3711896.3736872},
abstract = {Accurately forecasting the impact of macroeconomic events is critical for investors and policymakers. Salient events like monetary policy decisions and employment reports often trigger market movements by shaping expectations of economic growth and risk, thereby establishing causal relationships between events and market behavior. Existing forecasting methods typically focus either on textual analysis or time-series modeling, but fail to capture the multi-modal nature of financial markets and the causal relationship between events and price movements. To address these gaps, we propose CAMEF (Causal-Augmented Multi-Modality Event-Driven Financial Forecasting), a multi-modality framework that effectively integrates textual and time-series data with a causal learning mechanism and an LLM-based counterfactual event augmentation technique for causal-enhanced financial forecasting. Our contributions include: (1) a multi-modal framework that captures causal relationships between policy texts and historical price data; (2) a new financial dataset with six types of macroeconomic releases from 2008 to April 2024, and high-frequency real trading data for five key U.S. financial assets; and (3) an LLM-based counterfactual event augmentation strategy. We compare CAMEF to state-of-the-art transformer-based time-series and multi-modal baselines, and perform ablation studies to validate the effectiveness of the causal learning mechanism and event types.},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {3867–3878},
numpages = {12},
keywords = {causal learning, financial dataset, multimodal learning, time-series forecasting},
location = {Toronto ON, Canada},
series = {KDD '25}
}
