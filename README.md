# Imageomics HDR Anomaly Challenge Sample

This repository contains sample training code and submissions for the [2024 HDR Anomaly Challenge: Hybrid Butterfly Detection](https://www.codabench.org/competitions/3764/). It is designed to give participants a reference for both working on the challenge, and also the expected publication of their submissions following the challenge (i.e., how to open-source your submission).

## Repository Structure

For your repository, you will want to complete the structure information below and add other files (e.g., training code):
```
submission
  <model weights>
  model.py
  requirements.txt
```
We also recommend that you include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.

**Note:** If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/ingestion_program/whitelist.txt), please check the [issues](https://github.com/Imageomics/HDR-anomaly-challenge/issues) on the challenge GitHub to see if someone else has requested it before making your own issue.

### Structure of this Repository
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


This repository also includes `butterfly_sample_notebook.ipynb` which loads the metadata for the images and displays a histogram of the hybrid/non-hybrid distribution by subspecies. It then downloads 15% of the data and runs through a simplified sample submission training with that subset (the sample image amount can be adjusted to work within network constraints). To run this notebook, first clone this repository and create a fresh `conda` environment, then install the requirements file:
```
conda create -n butterfly-sample -c conda-forge pip -y
conda activate butterfly-sample
pip install -r requirements.txt
jupyter lab
```

## References
List any sources used in developing your model (e.g., baseline model that was fine-tuned).

[training data references](butterfly_anomaly.bib)
