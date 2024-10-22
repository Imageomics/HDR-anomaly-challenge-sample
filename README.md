# Imageomics HDR Anomaly Challenge Sample

This repository contains sample training code and submissions for the [2024 HDR Anomaly Challenge: Hybrid Butterfly Detection]()<!-- Add URL -->. It is designed to give participants a reference for both working on the challenge, and also the expected publication of their submissions following the challenge (i.e., how to open-source your submission).

## Repository Structure

<!-- 
complete structure info below
You may also wish to include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.
-->
```
HDR-anomaly-challenge-sample
│
├── BioCLIP_code_submission
│   ├── clf.pkl
│   ├── metadata
│   ├── model.py
│   └── requirements.txt
│
├── BioCLIP_train
│   ├── classifier.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── model_utils.py
│   └── training.py
│
├── DINO_SGD_code_submission
│   ├── clf.pkl
│   ├── metadata
│   ├── model.py
│   └── requirements.txt
│
└── DINO_train
    ├── classifier.py
    ├── data_utils.py
    ├── dataset.py
    ├── evaluation.py
    ├── model_utils.py
    └── training.py
```
**Note:** If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/ingestion_program/whitelist.txt), please check the [issues](https://github.com/Imageomics/HDR-anomaly-challenge/issues) on the challenge GitHub to see if someone else has requested it before making your own issue.

## References
<!-- list any sources (e.g., baseline model that was fine-tuned)
-->
[training data references](butterfly_anomaly.bib)
