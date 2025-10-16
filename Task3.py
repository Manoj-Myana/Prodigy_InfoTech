# Install dependencies
!pip install -q pandas scikit-learn matplotlib seaborn

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download the dataset
zip_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
zip_path = "bank-marketing.zip"
if not os.path.exists(zip_path):
    !wget -q {zip_url} -O {zip_path}
    print("✅ Downloaded main ZIP")

# Step 2: Extract outer ZIP
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("bank-marketing")
print("✅ Extracted main ZIP")

# Step 3: Extract nested ZIPs
nested_zips = ["bank.zip", "bank-additional.zip"]
for z in nested_zips:
    z_path = os.path.join("bank-marketing", z)
    if os.path.exists(z_path):
        with zipfile.ZipFile(z_path, "r") as nested_zip:
            nested_zip.extractall(os.path.join("bank-marketing", z.replace(".zip", "")))
        print(f"✅ Extracted nested ZIP: {z}")

# Step 4: Find CSV files
csv_files = []
for root, dirs, files in os.walk("bank-marketing"):
    for f in files:
        if f.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, f))

print("\n✅ Found CSV files:")
for f in csv_files:
    print(f)

# Step 5: Pick preferred CSV
preferred = ["bank-full.csv", "bank-additional-full.csv", "bank.csv"]
csv_path = None
for pref in preferred:
    for c in csv_files:
        if c.lower().endswith(pref.lower()):
            csv_path = c
            break
    if csv_path:
        break

if not csv_path:
    csv_path = csv_files[0]

print("\n➡️ Using CSV file:", csv_path)

# Step 6: Load dataset
df = pd.read_csv(csv_path, sep=';')
print("✅ Data loaded. Shape:", df.shape)
print(df.head())

# Step 7: Preprocessing
le_target = LabelEncoder()
df['y'] = le_target.fit_transform(df['y'])  # yes/no → 1/0

X = df.drop('y', axis=1)
y = df['y']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ✅ Updated for scikit-learn >= 1.2
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X), 0))
X_num = X[num_cols].values if num_cols else np.empty((len(X), 0))

X_all = np.hstack([X_num, X_cat])
feature_names = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=42, stratify=y
)

# Step 9: Train decision tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Step 10: Evaluation
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names,
          class_names=le_target.classes_,
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (depth=3 view)")
plt.show()
