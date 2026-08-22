# Hotel Booking Cancellation Prediction

Machine-learning project for exploring hotel booking data and predicting whether a reservation will be canceled.

## Quick Links

- [Project structure](#project-structure)
- [Installation](#installation)
- [Run the pipeline](#run-the-pipeline)
- [Explore the data](#explore-the-data)
- [Use the inference pipeline](#use-the-inference-pipeline)
- [Data files](#data-files)
- [Models and artifacts](#models-and-artifacts)

## Project Structure

| Path                                                             | Purpose                                   |
| ---------------------------------------------------------------- | ----------------------------------------- |
| [data/raw/](data/raw/)                                           | Original hotel booking dataset            |
| [data/preprocessed/](data/preprocessed/)                         | Train and test data prepared for modeling |
| [notebooks/EDA.ipynb](notebooks/EDA.ipynb)                       | Exploratory data analysis                 |
| [notebooks/baseline_model.ipynb](notebooks/baseline_model.ipynb) | Baseline model experiment                 |
| [notebooks/improved_model.ipynb](notebooks/improved_model.ipynb) | Improved model experiment                 |
| [src/data_loader.py](src/data_loader.py)                         | Data loading and train/test splitting     |
| [src/feature_engineering.py](src/feature_engineering.py)         | Feature creation, encoding, and scaling   |
| [src/improved_model.py](src/improved_model.py)                   | ExtraTrees training and evaluation        |
| [src/inference_pipeline.py](src/inference_pipeline.py)           | Single-record and batch predictions       |
| [scripts/feature_engineering.py](scripts/feature_engineering.py) | Preprocessing entry point                 |
| [scripts/improved_model.py](scripts/improved_model.py)           | Model training entry point                |
| [requirements.txt](requirements.txt)                             | Python dependencies                       |

## Installation

The project uses Python 3.12 and includes a virtual environment in `myenv/`.

```bash
source myenv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

Run commands from the project root:

```bash
python scripts/feature_engineering.py
python scripts/improved_model.py
```

The training script saves the improved model to [models/improved/](models/improved/) and preprocessing artifacts to [models/artifacts/](models/artifacts/).

## Explore the Data

Open [notebooks/EDA.ipynb](notebooks/EDA.ipynb) in Jupyter or VS Code. The model experiments are available in [notebooks/baseline_model.ipynb](notebooks/baseline_model.ipynb) and [notebooks/improved_model.ipynb](notebooks/improved_model.ipynb).

## Use the Inference Pipeline

After training, import [InferencePipeline](src/inference_pipeline.py) and provide a dictionary containing the raw booking fields:

```python
from src.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()
result = pipeline.predict_one(booking)
print(result)
```

The result contains `prediction`, `probability`, and the decision `threshold`.

## Data Files

- [Raw dataset](data/raw/hotel_bookings_updated_2024.csv)
- [Training dataset](data/preprocessed/train_preprocessed.csv)
- [Test dataset](data/preprocessed/test_preprocessed.csv)

The target column is `is_canceled`.

## Models and Artifacts

- [Baseline model](models/baseline/baseline_model.joblib)
- [Improved model](models/improved/)
- [Preprocessing artifacts](models/artifacts/)

Logs generated during preprocessing, training, and inference are stored in [logs/](logs/).
