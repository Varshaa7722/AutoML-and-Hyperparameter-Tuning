# model_selection.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models():
    """
    Returns uninitialized model instances for evaluation and tuning.
    Hyperparameter tuning logic (in hyperparameter_tuning.py) will
    handle XGBoost and RandomForest multiclass settings.
    """
    return {
        'logistic': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'lightgbm': LGBMClassifier()
    }
