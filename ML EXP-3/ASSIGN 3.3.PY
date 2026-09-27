import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [20, 22, 23, 25, 28, 30],
    "Salary": [25000, 30000, 28000, 30000, 45000, 50000],
    "Experience": [1, 2, 1, 3, 4, 5]
}
df = pd.DataFrame(data)

features = ["Age", "Salary", "Experience"]
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(df[features])

print("--- StandardScaler ---")
print(standard_data)
print("\nStandardScaler Minimum:")
print(standard_data.min())
print("StandardScaler Maximum:")
print(standard_data.max())

minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(df[features])

print("\n--- MinMaxScaler ---")
print(minmax_data)
print("\nMinMaxScaler Minimum:")
print(minmax_data.min())
print("MinMaxScaler Maximum:")
print(minmax_data.max())