# automl_pipeline.py
from src.feature_selection import select_features
from src.model_selection import get_models
from src.hyperparameter_tuning import tune_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def run_automl(X_train, X_val, y_train_raw, y_val_raw):
    # Step 0️⃣: Encode raw string labels into integer classes
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val   = le.transform(y_val_raw)

    # Step 1: Feature Selection on training data
    X_train_selected, mask = select_features(X_train, y_train, k=10)
    X_val_selected = X_val[:, mask]

    # Step 2: Load candidate models (uninitialized classes)
    models = get_models()
    best_model, best_acc = None, 0.0

    # Step 3: Train and evaluate each model
    for name, model_instance in models.items():
        print(f"🔍 Tuning {name}...")

        # Hyperparameter tuning expects model class, X and integer-encoded y
        best_params = tune_model(type(model_instance), X_train_selected, y_train)

        # Initialize tuned model
        model = type(model_instance)(**best_params, random_state=42)
        model.fit(X_train_selected, y_train)

        # Evaluate on validation data
        preds = model.predict(X_val_selected)
        acc = accuracy_score(y_val, preds)
        print(f"{name} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_model, best_acc = model, acc

    print(f"✅ Best validation accuracy: {best_acc:.4f} from model {best_model}")
    return best_model, le

