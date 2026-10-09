import os
from dotenv import load_dotenv
import snowflake.connector
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
)
df = pd.read_sql("SELECT * FROM hr_attrition_final;", conn)
conn.close()

# Drop ID column
data = df.drop(columns=['EMP_ID'])

# Encode categorical columns
label_encoders = {}
cat_cols = data.select_dtypes(include='object').columns.drop('ATTRITION')
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Target encoding
data['ATTRITION'] = data['ATTRITION'].map({'Yes': 1, 'No': 0})

X = data.drop(columns=['ATTRITION'])
y = data['ATTRITION']

# Fill remaining missing values with column median
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Balance training data with SMOTE ----
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_bal.value_counts().to_dict())

# ---- Hyperparameter tuning ----
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train_bal, y_train_bal)

model = grid.best_estimator_
print("\nBest params:", grid.best_params_)

# ---- Evaluate on original (untouched) test set ----
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop factors driving attrition:\n", importance.head(8))

# Save model + encoders for Streamlit later
with open("models/attrition_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": label_encoders, "columns": list(X.columns)}, f)

print("\nModel saved as attrition_model.pkl")