import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    "Age": [20, 22, None, 25, 28, 30],
    "Salary": [25000, 30000, 28000, None, 45000, 50000],
    "Department": ["IT", "HR", "IT", "Sales", "HR", None],
    "Experience": [1, 2, 1, 3, None, 5]
}
df = pd.DataFrame(data)
print("--- Original Dataset ---")
print(df)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Experience"] = df["Experience"].fillna(df["Experience"].median())
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print("\n--- Preprocessed Dataset ---")
print(df)

