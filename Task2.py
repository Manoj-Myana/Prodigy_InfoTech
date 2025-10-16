import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Task-02: Data Cleaning and Exploratory Data Analysis (EDA)
# ----------------------------------------------------

# Load the Titanic dataset
try:
    df_titanic = sns.load_dataset('titanic')
    print("Successfully loaded the Titanic dataset from seaborn.")
except Exception as e:
    print(f"Failed to load dataset: {e}. Using a synthetic, representative DataFrame for analysis.")
    # Fallback: Create a synthetic DataFrame that mirrors the necessary structure
    n_rows = 891
    np.random.seed(42)
    df_titanic = pd.DataFrame({
        'survived': np.random.choice([0, 1], n_rows, p=[0.62, 0.38]),
        'pclass': np.random.choice([1, 2, 3], n_rows, p=[0.25, 0.21, 0.54]),
        'sex': np.random.choice(['male', 'female'], n_rows, p=[0.65, 0.35]),
        'age': np.clip(np.random.normal(30, 15, n_rows), 1, 80),
        'fare': np.clip(np.random.lognormal(2.5, 1.5, n_rows), 5, 500),
        'embarked': np.random.choice(['S', 'C', 'Q', np.nan], n_rows, p=[0.72, 0.19, 0.08, 0.01]),
        'cabin': np.random.choice([f'C{i}' for i in range(1, 10)] + [np.nan], n_rows, p=[0.1]*9 + [0.1]),
    })
    df_titanic['age'] = df_titanic['age'].apply(lambda x: x if np.random.rand() > 0.2 else np.nan)


# --- 1. Data Cleaning and Preparation (ROBUST FIX) ---

# a) 'Age': Fill missing 'age' values with the median
median_age = df_titanic['age'].median()
df_titanic['age'] = df_titanic['age'].fillna(median_age)

# b) 'Embarked': Fill missing 'embarked' values with the mode
mode_embarked = df_titanic['embarked'].mode()[0]
df_titanic['embarked'] = df_titanic['embarked'].fillna(mode_embarked)

# c) 'Cabin': Create 'Has_Cabin' feature and drop 'cabin'
#    ***ROBUST FIX: Using try/except block for the 'cabin' column***
try:
    df_titanic['Has_Cabin'] = df_titanic['cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df_titanic.drop('cabin', axis=1, inplace=True, errors='ignore')
    print("Created 'Has_Cabin' feature and dropped 'cabin'.")
except KeyError:
    # If 'cabin' column doesn't exist, we skip feature creation and set default
    df_titanic['Has_Cabin'] = 0 # Assume no cabin info if column is missing
    print("Skipped 'Has_Cabin' feature creation: 'cabin' column not found.")


# d) Feature Engineering: Log-transformed Fare
df_titanic['Log_Fare'] = np.log1p(df_titanic['fare'])

# e) Convert 'survived' (0 or 1) to a descriptive label for plotting
df_titanic['survived_label'] = df_titanic['survived'].replace({0: 'Did Not Survive', 1: 'Survived'})

print("\nData Cleaning Complete. Starting EDA.")

# ----------------------------------------------------
# --- 2. Exploratory Data Analysis (EDA) Visualizations ---
# ----------------------------------------------------

# EDA 2.1: Survival vs. Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='survived_label', data=df_titanic, palette=['#D55E00', '#009E73'])
plt.title(' Survival Count by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Survival Status')
plt.savefig('task02_survival_by_gender.png')
plt.show()

# EDA 2.2: Survival vs. Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(x='pclass', hue='survived_label', data=df_titanic, order=[1, 2, 3], palette=['#D55E00', '#009E73'])
plt.title(' Survival Count by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class (1=Upper, 2=Middle, 3=Lower)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Survival Status')
plt.savefig('task02_survival_by_pclass.png')
plt.show()

# EDA 2.3: Age Distribution vs. Survival
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_titanic, x='age', hue='survived_label', fill=True, common_norm=False, palette=['#D55E00', '#009E73'], alpha=.5, linewidth=2)
plt.title(' Age Distribution by Survival Status', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.savefig('task02_age_vs_survival_kde.png')
plt.show()

# EDA 2.4: Age vs. Log-Transformed Fare with Survival Hue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='Log_Fare', hue='survived_label', data=df_titanic, palette=['#D55E00', '#009E73'], alpha=0.7)
plt.title(' Age vs. Log-Transformed Fare by Survival Status', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Log(Fare)', fontsize=14)
plt.legend(title='Survival Status')
plt.grid(axis='both', alpha=0.5)
plt.savefig('task02_age_vs_logfare_scatter.png')
plt.show()

# EDA 2.5: Survival Rate by Embarked Port
embarked_survival_rate = df_titanic.groupby('embarked')['survived'].apply(lambda x: (x == 1).sum() / len(x)).reset_index(name='Survival_Rate')

plt.figure(figsize=(8, 6))
sns.barplot(x='embarked', y='Survival_Rate', data=embarked_survival_rate, palette='viridis', order=['C', 'Q', 'S'])
plt.title(' Survival Rate by Embarkation Port', fontsize=16, fontweight='bold')
plt.xlabel('Embarkation Port (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=14)
plt.ylabel('Survival Rate', fontsize=14)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('task02_survival_by_embarked.png')
plt.show()
