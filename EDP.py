import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

df = pd.read_csv("student_performance_eda_dataset.csv")

os.makedirs("EDA_Images", exist_ok=True)

print(df.head())

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.info()

print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

print(df.describe())

print(df["Extracurricular_Level"].value_counts())
print(df["Performance_Level"].value_counts())

plt.figure(figsize=(8, 5))
sns.histplot(df["Exam_Score"], bins=10, kde=True)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.savefig("EDA_Images/01_exam_score_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["Study_Hours_Per_Day"], bins=10, kde=True)
plt.title("Distribution of Daily Study Hours")
plt.xlabel("Study Hours per Day")
plt.ylabel("Number of Students")
plt.savefig("EDA_Images/02_study_hours_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Study_Hours_Per_Day", y="Exam_Score")
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.savefig("EDA_Images/03_study_hours_vs_exam_score.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Attendance_Percentage", y="Exam_Score")
plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance Percentage")
plt.ylabel("Exam Score")
plt.savefig("EDA_Images/04_attendance_vs_exam_score.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Sleep_Hours_Per_Day", y="Exam_Score")
plt.title("Sleep Hours vs Exam Score")
plt.xlabel("Sleep Hours per Day")
plt.ylabel("Exam Score")
plt.savefig("EDA_Images/05_sleep_hours_vs_exam_score.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Assignments_Completed", y="Exam_Score")
plt.title("Assignments Completed vs Exam Score")
plt.xlabel("Assignments Completed")
plt.ylabel("Exam Score")
plt.savefig("EDA_Images/06_assignments_vs_exam_score.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Performance_Level")
plt.title("Student Performance Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")
plt.savefig("EDA_Images/07_performance_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Extracurricular_Level", y="Exam_Score")
plt.title("Extracurricular Level vs Exam Score")
plt.xlabel("Extracurricular Level")
plt.ylabel("Exam Score")
plt.savefig("EDA_Images/08_extracurricular_vs_exam_score.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

print(correlation)

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("EDA_Images/09_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

exam_correlation = correlation["Exam_Score"].sort_values(ascending=False)
print(exam_correlation)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[
        [
            "Study_Hours_Per_Day",
            "Attendance_Percentage",
            "Sleep_Hours_Per_Day",
            "Assignments_Completed",
            "Exam_Score"
        ]
    ]
)
plt.title("Outlier Detection using Boxplot")
plt.xticks(rotation=20)
plt.savefig("EDA_Images/10_outlier_detection.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

top_students = df.sort_values(
    by="Exam_Score",
    ascending=False
).head(10)

print(top_students)

avg_score = df.groupby("Extracurricular_Level")["Exam_Score"].mean()
print(avg_score)

plt.figure(figsize=(8, 5))
avg_score.plot(kind="bar")
plt.title("Average Exam Score by Extracurricular Level")
plt.xlabel("Extracurricular Level")
plt.ylabel("Average Exam Score")
plt.xticks(rotation=0)
plt.savefig("EDA_Images/11_average_score_extracurricular.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

performance_avg = df.groupby(
    "Performance_Level",
    observed=True
)["Exam_Score"].mean()

print(performance_avg)

print("Average Exam Score:", round(df["Exam_Score"].mean(), 2))
print("Average Study Hours:", round(df["Study_Hours_Per_Day"].mean(), 2))
print("Average Attendance:", round(df["Attendance_Percentage"].mean(), 2))
print("Average Sleep Hours:", round(df["Sleep_Hours_Per_Day"].mean(), 2))
print("Average Assignments Completed:", round(df["Assignments_Completed"].mean(), 2))
print("Highest Exam Score:", df["Exam_Score"].max())
print("Lowest Exam Score:", df["Exam_Score"].min())

print(
    df.select_dtypes(include=np.number)
    .corr()["Exam_Score"]
    .sort_values(ascending=False)
)
