import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df_titanic = sns.load_dataset('titanic')

# Display the first few rows and column information
print("--- Titanic Dataset Head (Relevant Columns) ---")
print(df_titanic[['age', 'sex', 'survived']].head())
print("\n--- Titanic Dataset Info ---")
print(df_titanic.info())

# --- Data Preparation for Age Histogram ---
# The 'age' column has missing values (NaN), which must be handled for a histogram.
# We'll fill NaN values with the median age to avoid losing data points.
df_titanic['age'].fillna(df_titanic['age'].median(), inplace=True)
print(f"\nMissing 'age' values filled with median: {df_titanic['age'].median()}")

# ----------------------------------------------------
# 1. Histogram for the Continuous Variable: Age
# ----------------------------------------------------

plt.figure(figsize=(10, 6))
# Create the histogram for Age
sns.histplot(df_titanic['age'], bins=20, kde=True, color='#0072B2', edgecolor='black')

plt.title('Distribution of Passenger Ages on the Titanic', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency (Number of Passengers)', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('titanic_age_histogram.png')
plt.show()
print("\nGenerated: titanic_age_histogram.png")

# ----------------------------------------------------
# 2. Bar Chart for the Categorical Variable: Gender (Sex)
# ----------------------------------------------------

# Count the occurrences of each gender
gender_counts = df_titanic['sex'].value_counts()

plt.figure(figsize=(8, 6))
# Create the bar chart for Gender (Sex)
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=['#D55E00', '#009E73'])

plt.title('Distribution of Passenger Gender (Sex) on the Titanic', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('titanic_gender_bar_chart.png')
plt.show()
print("Generated: titanic_gender_bar_chart.png")
