import pandas as pd

# Load the dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Display the first five records
print("\nFirst 5 student records:")
print(df.head())

# Display dataset information
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())