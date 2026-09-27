import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

data = {
    "Age": [20, 22, None, 25, 28, 30],
    "Salary": [25000, 30000, 28000, None, 45000, 50000],
    "Department": ["IT", "HR", "IT", "Sales", "HR", None],
    "Experience": [1, 2, 1, 3, None, 5]
}
df = pd.DataFrame(data)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Experience"] = df["Experience"].fillna(df["Experience"].median())
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

X = df[["Age", "Salary", "Department", "Experience"]]
numeric_features = ["Age", "Salary", "Experience"]
categorical_features = ["Department"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False), categorical_features)
    ]
)
X_processed = preprocessor.fit_transform(X)

print("--- MinMaxScaler Transformed Array ---")
print(X_processed)
print("\nMinimum:", X_processed.min())
print("Maximum:", X_processed.max())