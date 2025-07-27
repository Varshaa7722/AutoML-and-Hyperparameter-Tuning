from src.automl_pipeline import run_automl
from src.utils import load_and_preprocess_amazon_data

csv_path = r"data/cleaned_reviews.csv"
X_train, X_val, X_test, y_train, y_val, y_test,le = load_and_preprocess_amazon_data(csv_path)

if __name__ == "__main__":
    model = run_automl(X_train, X_val, y_train, y_val)  # Pass actual data
    print("✅ Best model trained successfully!")

