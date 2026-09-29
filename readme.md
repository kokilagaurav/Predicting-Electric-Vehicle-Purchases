# Predicting Electric Vehicle Purchases

> Understand EV purchase intent and turn model experiments into a reproducible prediction system.

This project predicts whether a customer will buy an electric vehicle using demographic, financial, commuting, vehicle, charging, and environmental features. It combines exploratory analysis, feature engineering, and model comparisons to generate purchase scores and submission files.

**Current stage:** The FastAPI inference service, browser frontend, and Docker startup are implemented. The reusable training and evaluation pipeline is still being developed.

## Quick start

### Docker

Build and run the complete application from the repository root:

```powershell
docker build -t ev-purchase-predictor .
docker run --rm -p 8000:8000 ev-purchase-predictor
```

Open [http://localhost:8000](http://localhost:8000). The container serves the frontend and API from the same URL.

### Local API and frontend

```powershell
conda create -n ev-purchase python=3.11
conda activate ev-purchase
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API also provides interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serves the prediction frontend |
| `/health` | `GET` | Returns service and model health |
| `/predict` | `POST` | Returns an EV purchase prediction |
| `/docs` | `GET` | Opens Swagger API documentation |

The `/predict` request includes customer age, income, commute distance, vehicle ownership, charging access, environmental concern, gender, city type, car type, home charging, range anxiety, and subsidy availability. `Range_Anxiety_Level` accepts `Low`, `Medium`, or `High`.

## Contents


## Purpose and goals

The project explores how factors such as charging access, commute distance, income, subsidies, environmental concern, and range anxiety relate to EV purchase decisions.

The goal is to develop the project from data exploration through model delivery:

1. Understand patterns in customer purchase behavior through exploratory analysis.
2. Build useful features and compare models using stratified validation and ROC-AUC.
3. Make preprocessing, training, evaluation, and submission generation reproducible.
4. Serve the selected model through a FastAPI application and a prediction interface.
5. Add data and pipeline versioning, automated checks, containerization, and monitoring.

## Progress

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Exploratory analysis | Available in notebooks | Feature distributions, target relationships, correlations, and paired feature analysis |
| Data preparation | Available in notebooks | Duplicate removal, feature exclusion, target encoding, and preprocessing |
| Feature engineering | Available in notebooks | Charging availability, subsidy interactions, commute ratios, and income interactions |
| Model experiments | Available in notebooks | XGBoost, CatBoost, Random Forest, hyperparameter search, and blending |
| Model interpretation | Available in notebooks | SHAP feature contribution analysis |
| Shared configuration | Implemented | Data paths and feature groups in `configs/data_configs.py`; model and metric paths in `configs/model_configs.py` |
| Loading and logging utilities | Implemented | CSV loading with error logging; dated log files and console output |
| Reusable ML pipeline | Scaffolded | The five `src/` modules currently contain responsibility outlines |
| DVC and parameters | Planned | `dvc.yaml` and `params.yaml` contain placeholder comments |
| Inference API and frontend | Implemented | FastAPI serves `/health`, `/predict`, Swagger docs, and the responsive frontend |
| Container startup | Implemented | Docker starts Uvicorn on port `8000` and serves the frontend from the same origin |
| Packaging and quality checks | In progress | Docker workflow is implemented; automated tests, CI/CD, and monitoring remain to be added |

## Modeling work and recorded results

The main experimental reference is [`notebook/model2_fixed.ipynb`](notebook/model2_fixed.ipynb). Earlier exploration is preserved in [`model1.ipynb`](notebook/model1.ipynb) and [`model2.ipynb`](notebook/model2.ipynb).

Work covered so far includes:


The following values are **saved notebook outputs**, not results from a fresh run or an automated benchmark:

| Experiment | Evaluation | Recorded ROC-AUC |
| --- | --- | --- |
| Initial XGBoost pipeline in `model2_fixed.ipynb` | Mean five-fold CV | 0.93898 |
| Tuned XGBoost | Holdout split | 0.94265 |
| XGBoost with additional engineered features | Mean five-fold CV | 0.94180 |
| CatBoost | Out-of-fold predictions | 0.94172 |
| Random Forest | Mean five-fold CV | 0.94008 |

These rows use different evaluation procedures and feature sets, so they do not establish a final model ranking. A consistent evaluation and model selection process remains part of the pipeline milestone.

## Dataset

Place the expected CSV files under `data/raw/`:

| File | Purpose |
| --- | --- |
| `train.csv` | Labeled examples for training and validation |
| `test.csv` | Unlabeled examples for submission predictions |
| `sample_submission.csv` | Reference output columns and test IDs |

**CSV files are ignored by Git.** They are present in the current local workspace, but a fresh clone needs the dataset supplied separately.


**Configuration follow-up:** The raw `Range_Anxiety_Level` field contains text labels, such as `Low`, while `DataConfig` currently lists it as numerical. Align that configuration with the data before using it to build the reusable preprocessing pipeline.

## Repository structure

```text
.
|-- configs/
|   |-- data_configs.py       # Dataset paths, target, ID, and feature groups
|   `-- model_configs.py      # Default model name and artifact paths
|-- data/                     # Local CSV data; ignored by Git
|   |-- raw/                  # Training, test, and sample submission files
|   |-- processed/            # Reserved for processed datasets
|   `-- submission-files/     # Existing local submission candidates
|-- notebook/
|   |-- model1.ipynb          # Initial exploration and modeling
|   |-- model2.ipynb          # Follow-up exploration
|   |-- model2_fixed.ipynb    # Feature engineering and model comparisons
|   `-- catboost_info/        # Saved CatBoost training logs
|-- src/                      # Pipeline module outlines
|   |-- data_ingestion.py
|   |-- data_manipulation.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   `-- model_evaluation.py
|-- utils/
|   |-- data_loader.py        # CSV loading with logging and error handling
|   `-- logger.py             # File and console logging
|-- models/                   # Reserved for serialized models
|-- frontend/                 # Prediction form and responsive styles
|-- main.py                   # FastAPI API and frontend server
|-- dvc.yaml                  # Placeholder pipeline definition
|-- params.yaml               # Placeholder experiment parameters
|-- Dockerfile                # Uvicorn container startup
|-- requirements.txt          # Core dependencies
`-- LICENSE                   # MIT license
```

## Notebook usage

The notebooks contain the exploratory analysis and model experiments. After installing the API dependencies, install the additional notebook packages:

```powershell
conda create -n ev-purchase python=3.10
conda activate ev-purchase
python -m pip install -r requirements.txt
python -m pip install notebook matplotlib seaborn shap scipy
```

The second install command supplies notebook and analysis dependencies that are not explicitly listed in `requirements.txt`. Dependency versions are not pinned yet, so exact environment reproduction remains a follow-up.

After placing the dataset in `data/raw/`, launch Jupyter from the notebook directory:

```powershell
cd notebook
jupyter notebook model2_fixed.ipynb
```

Before running the notebook, update its CSV loading cell to match the current directory structure:

```python
df = pd.read_csv("../data/raw/train.csv")
df_t = pd.read_csv("../data/raw/test.csv")
```

The notebooks currently use `../data/train.csv` and `../data/test.csv`. Their output paths also point to `../data/`; update submission destinations to `../data/submission-files/<filename>.csv` to keep generated files in the intended folder.

Run cells in order. Cross-validation, hyperparameter search, and SHAP analysis can take substantial time on the full dataset.

The notebook experiments are separate from the currently deployed inference path. The API loads the saved artifacts from `models/best_model.pkl` and `models/label_encoder.pkl`.

## Pipeline design

The intended reusable workflow is:

```text
Raw CSV files
    |
    v
Data ingestion and validation
    |
    v
Duplicate removal and feature selection
    |
    v
Feature engineering
    |
    v
Preprocessing and model training
    |
    v
Evaluation, tuning, and model selection
    |
    +--> Submission files
    |
    v
Saved preprocessing, model, and metrics
    |
    v
FastAPI prediction service and frontend
```

| Module | Intended responsibility |
| --- | --- |
| `src/data_ingestion.py` | Load datasets through shared configuration and utilities |
| `src/data_manipulation.py` | Remove duplicates and drop excluded features while preserving output IDs |
| `src/feature_engineering.py` | Apply the same feature creation logic to training and inference inputs |
| `src/model_training.py` | Build preprocessing and model pipelines, handle missing values, split data, and train |
| `src/model_evaluation.py` | Evaluate metrics, cross-validate, tune models, and create submissions |

Preprocessing should be fitted within each training split and saved with the selected model so inference uses the same transformations. DVC stages and YAML parameters will coordinate the workflow once these modules are executable.

## Roadmap

### Next milestone: reproducible training

- [ ] Consolidate ingestion, manipulation, feature engineering, training, and evaluation into one executable command.
- [ ] Add required-column and missing-value validation.
- [ ] Pin dependency versions and add focused pipeline tests.


### Following milestone: versioning and serving

- [ ] Complete the DVC stages and parameterized experiments.
- [x] Implement FastAPI prediction and health endpoints with input validation.
- [x] Connect the frontend form to the prediction API.


### Later milestone: deployment and maintenance

- [x] Implement Docker build and service startup.
- [ ] Add CI checks, deployment automation, and model monitoring.


## Outputs

| Output | Current state |
| --- | --- |
| `data/submission-files/*.csv` | Local notebook-generated submission candidates |
| `notebook/catboost_info/` | Saved CatBoost experiment logs |
| `logs/YYYY-MM-DD.log` | Created when the shared logging utility is used |
| `data/processed/train_processed.csv` and `test_processed.csv` | Configured destinations; generation is planned |
| `models/best_model.pkl` and `models/label_encoder.pkl` | Loaded by the running FastAPI service |
| `artifacts/metrics.json` | Configured metrics destination; automated export is planned |

Submission files follow this schema:

```text
id,Will_Buy_EV
```

Preserve the test-set IDs and use `sample_submission.csv` as the format reference. Probability-based submissions contain scores for the positive purchase class; rank-blended submissions contain normalized ranking scores, which should not be interpreted as calibrated purchase probabilities.

## License

This project is available under the [MIT License](LICENSE).
