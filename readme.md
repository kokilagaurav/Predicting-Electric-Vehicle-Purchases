# Predicting Electric Vehicle Purchases

> Understand EV purchase intent and turn model experiments into a reproducible prediction system.

This project predicts whether a customer will buy an electric vehicle using demographic, financial, commuting, vehicle, charging, and environmental features. It combines exploratory analysis, feature engineering, and model comparisons to generate purchase scores and submission files.

**Current stage:** Notebook experiments are available. The next milestone is to move that work into a reusable Python pipeline; the API, frontend, and deployment workflow are planned.

## Contents

- [Purpose and goals](#purpose-and-goals)
- [Progress](#progress)
- [Modeling work and recorded results](#modeling-work-and-recorded-results)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Setup and notebook usage](#setup-and-notebook-usage)
- [Pipeline design](#pipeline-design)
- [Roadmap](#roadmap)
- [Outputs](#outputs)
- [License](#license)

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
| Inference API and frontend | Planned | `main.py` and frontend files are placeholders |
| Packaging and quality checks | Planned | Dockerfile is a placeholder; automated tests, CI/CD, and monitoring remain to be added |

## Modeling work and recorded results

The main experimental reference is [`notebook/model2_fixed.ipynb`](notebook/model2_fixed.ipynb). Earlier exploration is preserved in [`model1.ipynb`](notebook/model1.ipynb) and [`model2.ipynb`](notebook/model2.ipynb).

Work covered so far includes:

- Numerical and categorical analysis, including skewness and relationships with the target.
- Engineered features such as total charging availability, income multiplied by subsidy availability, charging availability per commute distance, and income per car.
- Scaling, log transformations for selected numerical columns, and one-hot encoding.
- SMOTE in the initial notebook; class weighting in the later XGBoost, CatBoost, and Random Forest experiments.
- Five-fold stratified cross-validation, a stratified holdout split, and XGBoost tuning with `RandomizedSearchCV`.
- CatBoost out-of-fold predictions, weighted probability blending, rank blending, and SHAP analysis.

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

- **Target:** `Will_Buy_EV`, stored as `Yes` / `No` in the raw training data.
- **Identifier:** `id`, retained for output and excluded from model inputs.
- **Numerical inputs:** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, and `Environmental_Concern_Level`.
- **Categorical inputs:** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, and `Range_Anxiety_Level`.

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
|-- frontend/                 # Placeholder HTML and CSS
|-- main.py                   # Placeholder FastAPI entrypoint
|-- dvc.yaml                  # Placeholder pipeline definition
|-- params.yaml               # Placeholder experiment parameters
|-- Dockerfile                # Placeholder container definition
|-- requirements.txt          # Core dependencies
`-- LICENSE                   # MIT license
```

## Setup and notebook usage

The current workflow runs through Jupyter notebooks. From the repository root, create an environment, for example with Conda:

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

There is no executable training command or FastAPI endpoint yet; `main.py` contains only a placeholder comment.

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

- [ ] Align feature configuration with the raw schema, including `Range_Anxiety_Level`.
- [ ] Implement ingestion and validation for required columns, missing values, and target labels.
- [ ] Move duplicate removal, feature selection, and feature creation into `src/`.
- [ ] Build a shared preprocessing and training pipeline with stratified validation.
- [ ] Add a training command that produces model, metrics, and submission artifacts.
- [ ] Pin dependencies and add focused data and pipeline tests.

### Following milestone: versioning and serving

- [ ] Define experiment settings in `params.yaml` and executable stages in `dvc.yaml`.
- [ ] Track datasets, model versions, and experiment metrics.
- [ ] Implement FastAPI prediction and health endpoints with input validation.
- [ ] Connect the frontend form to the prediction API.

### Later milestone: deployment and maintenance

- [ ] Implement the Docker build and service startup.
- [ ] Add CI checks for the pipeline and API.
- [ ] Add deployment automation, prediction logging, and data/model monitoring.

## Outputs

| Output | Current state |
| --- | --- |
| `data/submission-files/*.csv` | Local notebook-generated submission candidates |
| `notebook/catboost_info/` | Saved CatBoost experiment logs |
| `logs/YYYY-MM-DD.log` | Created when the shared logging utility is used |
| `data/processed/train_processed.csv` and `test_processed.csv` | Configured destinations; generation is planned |
| `models/model.pkl` | Configured model destination; automated export is planned |
| `artifacts/metrics.json` | Configured metrics destination; automated export is planned |

Submission files follow this schema:

```text
id,Will_Buy_EV
```

Preserve the test-set IDs and use `sample_submission.csv` as the format reference. Probability-based submissions contain scores for the positive purchase class; rank-blended submissions contain normalized ranking scores, which should not be interpreted as calibrated purchase probabilities.

## License

This project is available under the [MIT License](LICENSE).
