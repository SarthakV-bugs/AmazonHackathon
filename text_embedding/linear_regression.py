import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Load and Clean Data
# ============================================
df = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\Hackathon\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv")

def clean_extract(text):
    text = str(text)
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(value_match.group(1)) if value_match else 0
    unit_match = re.search(r"Unit:\s*([A-Za-z ]+)", text)
    unit = unit_match.group(1).strip() if unit_match else None
    text = re.sub(r"Value:\s*[\d\.]+", "", text)
    text = re.sub(r"Unit:\s*[A-Za-z ]+", "", text)
    text = re.sub(r"[\n\r\t\xa0]+", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return pd.Series([text, value, unit])

df[["clean_text", "Value", "Unit"]] = df["catalog_content"].apply(clean_extract)

# ============================================
# STEP 2: Extract Pack Count
# ============================================
PACK_PATTERNS = [
    r"(?:pack|case)\s*of\s*(\d+)",
    r"(\d+)\s*-\s*pack",
    r"(\d+)\s*count",
    r"\((\d+)\s*count\)",
    r"\((\d+)\s*ct\)",
]

def extract_pack_count(text):
    s = str(text).lower()
    for p in PACK_PATTERNS:
        m = re.search(p, s)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    return 1

df["pack_count"] = df["clean_text"].apply(extract_pack_count)

# ============================================
# STEP 3: Normalize Quantities
# ============================================
WEIGHT_UNITS = {
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "pound": 453.59, "pounds": 453.59,
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0,
}
VOLUME_UNITS = {
    "fl oz": 29.57, "floz": 29.57, "fluid ounce": 29.57,
    "ml": 1.0, "l": 1000.0,
}
COUNT_TOKENS = {"count", "ct", "each", "ea"}

def normalize_quantity(value, unit):
    if unit is None or pd.isna(unit):
        return 0.0, 0.0, 0.0
    u = str(unit).strip().lower()
    if u in WEIGHT_UNITS:
        return float(value) * WEIGHT_UNITS[u], 0.0, 0.0
    if u in VOLUME_UNITS:
        return 0.0, float(value) * VOLUME_UNITS[u], 0.0
    if u in COUNT_TOKENS or u == "count":
        return 0.0, 0.0, float(value)
    return 0.0, 0.0, 0.0

w_g, v_ml, p_ct = [], [], []
for _, row in df.iterrows():
    wg, vm, ct = normalize_quantity(row["Value"], row["Unit"])
    w_g.append(wg)
    v_ml.append(vm)
    p_ct.append(ct)

df["total_weight_g"] = w_g
df["total_volume_ml"] = v_ml
df["total_piece_ct"] = p_ct

# ============================================
# STEP 4: Prepare Features and Target
# ============================================
# Only numeric features - no text
X = df[["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]].values
y = df["price"].values

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# ============================================
# STEP 5: Train/Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# STEP 6: Linear Regression
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
# STEP 7: Predictions
# ============================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ============================================
# STEP 8: Evaluation
# ============================================
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

print("\n=== Training Results ===")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
print(f"Train SMAPE: {smape(y_train, y_train_pred):.2f}%")
print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")

print("\n=== Test Results ===")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Test SMAPE: {smape(y_test, y_test_pred):.2f}%")
print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")

# ============================================
# STEP 9: Show Coefficients
# ============================================
print("\n=== Model Coefficients ===")
feature_names = ["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:20s}: {coef:10.4f}")
print(f"{'Intercept':20s}: {model.intercept_:10.4f}")

# ============================================
# STEP 10: Visualization (optional)
# ============================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Actual vs Predicted Price')
plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=100)
print("\nPlot saved as 'linear_regression_results.png'")
