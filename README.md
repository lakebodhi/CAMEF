# 🌐 CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting

**CAMEF** is a novel framework for forecasting financial market trends by integrating **time-series data** and **macroeconomic policy texts**, enhanced by **causal learning through LLM-generated counterfactuals**.  
It was accepted at **KDD 2025**.

- 📄 [arXiv Paper](https://arxiv.org/abs/2502.04592)
- 📊 [Poster](https://github.com/lakebodhi/CAMEF/blob/main/resources/CAMEF_Poster.pdf)
- 📊 [Slides](https://github.com/lakebodhi/CAMEF/blob/main/resources/CAMEF_Slides.pdf)
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
    booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
    pages = {3867–3878},
    numpages = {12},
    keywords = {causal learning, financial dataset, multimodal learning, time-series forecasting},
    location = {Toronto ON, Canada},
    series = {KDD '25}
  }


## 📦 Dataset Access

We provide the processed **CAMEF dataset**, which includes both **raw** and **cleaned** versions of key macroeconomic announcements and high-frequency financial time-series data spanning from **2004 to 2024**.

### 📰 Macroeconomic Policy Texts

Six types of macroeconomic announcements:

- FOMC Minutes  
- Unemployment Insurance Claims  
- Employment Situation Reports  
- GDP Advance Releases  
- Consumer Price Index (CPI) Reports  
- Producer Price Index (PPI) Reports  

### 📈 Time-Series Financial Data

High-frequency (5-minute level) trading data for five major U.S. financial assets:

- S&P 500 Index  
- NASDAQ 100 Index  
- Dow Jones Industrial Average (DJIA)  
- U.S. 1-Month Treasury Yield  
- U.S. 10‑Year Treasury Yield  

### Generated Counterfactual Policy Texts:

We leveraged LLAMA-3 model with design prompt to generate 10 counterfactual events for each policy text by varying the sentiment level

📁 [Download the dataset from Google Drive](https://drive.google.com/your-dataset-link)  
📝 **Source**: All data are collected from publicly available government and financial databases. The full data preprocessing pipeline is included in this repository under `/data_preprocessing/`.

The folder structure is as the following 

