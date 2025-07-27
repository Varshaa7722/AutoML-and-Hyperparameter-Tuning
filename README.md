#  AutoML and Hyperparameter Optimization: Amazon Reviews Sentiment Classification

Automatic sentiment classification of Amazon product reviews into **positive**, **neutral**, or **negative**, using AutoML workflows, feature selection, and hyperparameter tuning.



## Project Overview

- **Objective**: Predict sentiment labels from Amazon review text and metadata using an automated pipeline that selects features, tunes models with Optuna, and evaluates performance.
- **Features used**: TF‑IDF features from review text, standardized numerical features (`review_score`, `cleaned_review_length`), with top‑10 feature selection via `SelectKBest`.
- **Models evaluated**: Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Tuning**: Optuna-based tuning for XGBoost and RandomForest using multiclass accuracy.


## Working

1. **Data Loading & Preprocessing** (`utils.py`)
   - Reads CSV, drops missing reviews, applies TF-IDF + standard scaling on numeric columns.
   - Splits data into stratified train/validation/test sets.
   - Encodes string sentiment labels (`negative`/`neutral`/`positive`) into integer classes (0, 1, 2) using `LabelEncoder`—required for XGBoost and LightGBM.

2. **Feature Selection** (`feature_selection.py`)
   - Selects top‑k features using `SelectKBest` with `f_classif`.
   - Applies same mask across train and validation/test sets.

3. **Model Setup** (`model_selection.py`)
   - Returns four initialized model objects:
     - `LogisticRegression`
     - `RandomForestClassifier`
     - `XGBClassifier` (configured with `use_label_encoder=False`, `eval_metric='mlogloss'`)
     - `LGBMClassifier`

4. **Hyperparameter Tuning** (`hyperparameter_tuning.py`)
   - Uses Optuna to tune hyperparameters:
     - For XGB: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `n_estimators`, `objective='multi:softprob'`, `num_class` set dynamically.
     - For RF: `n_estimators`, `max_depth`.
   - Evaluates via 3‑fold cross‑validation accuracy.

5. **Pipeline Execution** (`automl_pipeline.py`)
   - Feature selection → tuning → final model training → validation & model selection.




## Results
Class distribution: ~9,503 positive, ~6,300 neutral, ~1,534 negative reviews.

Validation accuracy reported per model: **XGBoost typically achieves highest.**






