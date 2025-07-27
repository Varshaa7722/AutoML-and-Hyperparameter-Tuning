# utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_amazon_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['cleaned_review'])
    print(df['sentiments'].value_counts())

    X = df[['cleaned_review', 'cleaned_review_length', 'review_score']]
    y = df['sentiments']

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000), 'cleaned_review'),
        ('num', StandardScaler(), ['cleaned_review_length', 'review_score'])
    ])

    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp_raw, test_size=0.5, stratify=y_temp_raw, random_state=42)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Label encoding: convert string labels to integer classes [0..n‑1]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, le


