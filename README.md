# Brain Tumor MRI Detection and Classification

This project is a COMP 263 deep learning application for detecting and classifying brain tumors from MRI scans using the Kaggle brain tumor dataset.

## Problem
Classify each MRI image into one of four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Recommended Model
The best current model in this repository is `EfficientNetB0` with transfer learning.

Why this is the best fit for this dataset:

- The dataset is image-based and relatively small, so transfer learning is more reliable than training a deep CNN from scratch.
- EfficientNetB0 gives a strong accuracy-efficiency tradeoff for local training and demos.
- It generalizes better than the baseline CNN on the saved checkpoints in this repo.

Current evaluated results on the provided test split:

- `EfficientNetB0 Transfer`: Accuracy `84.63%`, Macro F1 `84.11%`
- `Baseline CNN`: Accuracy `75.63%`, Macro F1 `73.53%`

An additional `DenseNet121` transfer model is included as a strong medical-imaging alternative for future experiments.

## Dataset
Place the Kaggle dataset in:

`data/brain_tumor_dataset/`

Expected structure:

```text
data/
└── brain_tumor_dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

## Features

- Baseline CNN and transfer-learning models
- Model comparison pipeline
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation
- Streamlit web app for MRI upload and prediction
- Saved plots and JSON metrics for the final report

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Train the recommended model:

```bash
python train.py --model efficientnet_b0_transfer
```

Train all supported models:

```bash
python train.py --model all
```

## Evaluate

Evaluate the recommended model:

```bash
python evaluate.py --model efficientnet_b0_transfer
```

Evaluate every trained model and build a leaderboard:

```bash
python evaluate.py --model all
```

## Run the App

```bash
streamlit run app.py
```

## Project Note
This application is for educational and research purposes only. It must not be used for real medical diagnosis.
