# import libabaries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# loadeded dataset 
df = pd.read_csv("retail_sales_dataset.csv")
print("Data loaded sucessfully!")
print(df.head())

# qucik look 
print("\n=== INFO() ===")
print(df.info())

print("\n=== Describe() ===")
print(df.describe())

print("\n=== Missing values count() ===")
print(df.isnull().sum())

# Step 3. Duplicates check and optional drop 
dup_count = df.duplicated().sum()
print("\n=== Duplicated rows: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates dropped.")

# Try to standardize typical columns if present
# Common retail columns: 'order_id', 'date', 'product', 'category', 'price', 'quantity', 'sales', 'customer_id'

# Parse date if exists
date_cols = [c for c in df.columns if "date" in c.lower()]
if date_cols:
    dcol = date_cols[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    print(f"Parsed datetime column: {dcol}")

# Derive 'sales' if missing and price * quantity available
col_lower = {c.lower(): c for c in df.columns}
price_col = col_lower.get("price")
qty_col = col_lower.get("quantity")
sales_col = col_lower.get("sales")

if sales_col is None and price_col and qty_col:
    df["sales"] = df[price_col].astype(float) * df[qty_col].astype(float)
    print("Derived 'sales' = price * quantity.")
elif sales_col:
    # Ensure numeric
    df["sales"] = pd.to_numeric(df[sales_col], errors="coerce")
else:
    # If none, create sales from any numeric columns as a fallback (sum of numerics)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df["sales"] = df[num_cols].sum(axis=1)
        print("No price/quantity found. Created 'sales' as sum of numeric columns (fallback).")
    else:
        # Create dummy sales to keep pipeline running
        df["sales"] = 0.0
        print("No numeric columns found. Created dummy 'sales' = 0.0.")

# Choose a category-like column for grouping
possible_cat_cols = ["category", "product_category", "segment", "department"]
cat_col = next((c for c in df.columns if c.lower() in possible_cat_cols), None)

if cat_col is None:
    # Try any object column with moderate cardinality
    obj_cols = df.select_dtypes(include=["object"]).columns
    cat_col = next((c for c in obj_cols if df[c].nunique() <= 50), None)

if cat_col is None:
    # Final fallback: create a synthetic category
    cat_col = "category_synthetic"
    df[cat_col] = "General"
    print("No suitable category found. Created synthetic 'category_synthetic'.")

grouped_sales = df.groupby(cat_col, dropna=False)["sales"].sum().sort_values(ascending=False)
print("\n=== Sales by category (grouped) ===")
print(grouped_sales.head(10))

# 1) Bar chart of aggreted sale by category 
'''plt.figure(figsize=(10, 8))
top_k = grouped_sales.head(10)
sns.barplot(x=top_k.index, y=top_k.values, palette="viridis")
plt.title(f"Top categories by sale ({cat_col})")
plt.xlabel(cat_col)
plt.ylabel("Total sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()'''

# 2) Histogram / distribution of sales 
'''plt.figure(figsize=(10, 6))
sns.histplot(df["sales"].dropna(), bins=30, kde=True, color="steelblue")
plt.title("Sales distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Create target (high vs low sales)
sales_median = df["sales"].median()
df["target_high_sales"] = (df["sales"] >= sales_median).astype(int)
print(f"\nTarget created using median sales = {sales_median:.2f}")

# 2) Feature selection
# Use all columns except obvious keys/target/sales; keep common retail features
exclude_cols = {"sales", "target_high_sales"}
key_like = {"order_id", "invoice_id", "id"}
exclude_cols |= {c for c in df.columns if c.lower() in key_like}

X = df.drop(columns=[c for c in df.columns if c in exclude_cols])
y = df["target_high_sales"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# If no features remain, create a synthetic one from sales noise
if len(num_cols) == 0 and len(cat_cols) == 0:
    X = pd.DataFrame({"synthetic_feature": df["sales"] + np.random.normal(0, 1, size=len(df))})
    num_cols = ["synthetic_feature"]
    cat_cols = []

print("\n=== Feature columns ===")
print(f"Numeric: {num_cols}")
print(f"Categorical: {cat_cols}")

# 3) Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# 4) Model: RandomForest (robust baseline)
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# 5) Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 6) Fit and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, digits=4))

# 7) Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred Low", "Pred High"],
            yticklabels=["True Low", "True High"])
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
