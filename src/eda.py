
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# 1. Basic dataset information
print("Dataset shape:", df.shape)
print("\nStatistical summary:")
print(df[["age", "studytime", "absences", "G1", "G2", "G3"]].describe())

# 2. Final grade distribution
print("\nFinal grade distribution:")
print(df["G3"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=range(0, 22), edgecolor="black")
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.xticks(range(0, 21, 2))
plt.tight_layout()
plt.savefig("grade_distribution.png")
plt.show()

# 3. Study time vs final grade
study_performance = df.groupby("studytime")["G3"].mean()

print("\nAverage final grade by study time:")
print(study_performance)

study_performance.plot(kind="bar", figsize=(8, 5))
plt.title("Study Time vs Average Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("studytime_vs_grade.png")
plt.show()

# 4. Correlation between grades
print("\nCorrelation between G1, G2 and G3:")
print(df[["G1", "G2", "G3"]].corr())

print("\nEDA completed successfully!")
