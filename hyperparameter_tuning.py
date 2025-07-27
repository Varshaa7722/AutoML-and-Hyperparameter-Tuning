# hyperparameter_tuning.py
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def objective(trial, model_class, X, y):
    # Random Forest tuning
    if model_class.__name__ == "RandomForestClassifier":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 32)
        }
        model = RandomForestClassifier(**params, random_state=42)
    # XGBoost tuning
    elif model_class.__name__ in ("XGBClassifier",):
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': len(set(y)),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0)
        }
        model = XGBClassifier(**params, use_label_encoder=False, random_state=42)
    else:
        # Default: no hyperparameters tuned
        model = model_class(random_state=42)

    # Evaluate using cross-validation
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

def tune_model(model_class, X, y, n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_class, X, y),
                   n_trials=n_trials)
    print(f"✔️ Best params for {model_class.__name__}: {study.best_params}")
    return study.best_params
