# Brain Tumor MRI Detector

This repository contains a complete COMP 263 deep learning project for brain tumor MRI image classification.

The repository already includes:

- the MRI dataset in `data/brain_tumor_dataset/`
- the trained model files in `models/`
- the saved evaluation outputs in `outputs/`
- the Streamlit application in `app.py`

Students can clone the repository and run the app directly after installing the Python dependencies.

## Classes

The model predicts one of these four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Included Best Model

The main recommended model in this project is:

- `EfficientNetB0 Transfer`

Saved evaluation result:

- Accuracy: `84.63%`
- Macro Precision: `84.84%`
- Macro Recall: `84.63%`
- Macro F1: `84.11%`

Baseline comparison:

- `Baseline CNN` Accuracy: `75.63%`
- `Baseline CNN` Macro F1: `73.53%`

## Quick Start

Open a terminal in the project folder and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Easiest Windows Setup

If you are using Windows, you can simply:

1. Run `setup.bat` once
2. Run `run_app.bat` to start the app

This lets students open the project and run it more easily without typing all commands manually.

## Project Structure

```text
COMP263-Brain-Tumor-Classification-main/
├── app.py
├── config.py
├── data/
│   └── brain_tumor_dataset/
├── models/
├── outputs/
├── train.py
├── evaluate.py
├── model_factory.py
├── data_utils.py
├── requirements.txt
├── setup.bat
├── run_app.bat
└── README.md
```

## Notes for Students

- The trained models are already included.
- You do not need to retrain the model to use the app.
- If you only want to test predictions, just install dependencies and run `streamlit run app.py`.
- Retraining is optional and only needed if you want to experiment with different models.

## Optional Commands

Run evaluation:

```powershell
python evaluate.py --model all
```

Retrain the recommended model:

```powershell
python train.py --model efficientnet_b0_transfer
```

## Important Note

This project is for educational use only and is not a medical diagnosis tool.
