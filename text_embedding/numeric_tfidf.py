import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ============================================
# STEP 1: Load and Clean Data
# ============================================
df = pd.read_csv(
    r"C:\Users\ASUS\PycharmProjects\Hackathon\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv")
df_copy = df.copy()


def clean_extract(text):
    text = str(text)

    # Extract Value
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(value_match.group(1)) if value_match else 0

    # Extract Unit
    unit_match = re.search(r"Unit:\s*([A-Za-z ]+)", text)
    unit = unit_match.group(1).strip() if unit_match else None

    # Remove Value and Unit lines
    text = re.sub(r"Value:\s*[\d\.]+", "", text)
    text = re.sub(r"Unit:\s*[A-Za-z ]+", "", text)

    # Remove Item Name header
    text = re.sub(r"Item Name:\s*", "", text)

    # Remove Bullet Point headers
    text = re.sub(r"Bullet Point\s*\d*:?\s*", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"[\n\r\t\xa0]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

    return pd.Series([text, value, unit])


df_copy[["clean_text", "Value", "Unit"]] = df_copy["catalog_content"].apply(clean_extract)

# ============================================
# STEP 2: Extract Pack Count and Normalize Quantities
# ============================================

# Pack extraction patterns
PACK_PATTERNS = [
    r"(?:pack|case)\s*of\s*(\d+)",
    r"(\d+)\s*-\s*pack",
    r"(\d+)\s*count",
    r"\((\d+)\s*count\)",
    r"\((\d+)\s*ct\)",
    r"(\d+)\s*ct\b",
    r"\(\s*pack\s*of\s*(\d+)\s*\)",
    r"\b(\d+)\s*each\b",
    r"\b(\d+)\s*ea\b",
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


# Normalization dicts
WEIGHT_UNITS = {
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "pound": 453.59, "pounds": 453.59,
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
}
VOLUME_UNITS = {
    "fl oz": 29.57, "floz": 29.57, "fluid ounce": 29.57, "fluid ounces": 29.57,
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
}
COUNT_TOKENS = {"count", "ct", "each", "ea"}


def normalize_quantity(value, unit):
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return 0.0, 0.0, 0.0
    u = str(unit).strip().lower()
    u = u.replace("ounces", "ounce").replace("lbs", "lb").replace("grams", "gram")
    if u in WEIGHT_UNITS:
        return float(value) * WEIGHT_UNITS[u], 0.0, 0.0
    if u in VOLUME_UNITS:
        return 0.0, float(value) * VOLUME_UNITS[u], 0.0
    if u in COUNT_TOKENS or u == "count":
        return 0.0, 0.0, float(value)
    return 0.0, 0.0, 0.0


# Apply pack extraction
df_copy["pack_count"] = df_copy["clean_text"].apply(extract_pack_count)

# Normalize quantities
w_g, v_ml, p_ct = [], [], []
for _, row in df_copy.iterrows():
    v = row.get("Value", 0)
    u = row.get("Unit", None)
    wg, vm, ct = normalize_quantity(v, u)
    w_g.append(wg)
    v_ml.append(vm)
    p_ct.append(ct)

df_copy["total_weight_g"] = w_g
df_copy["total_volume_ml"] = v_ml
df_copy["total_piece_ct"] = p_ct

# ============================================
# STEP 3: Feature Engineering - Numeric Features
# ============================================
numeric_features = ["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]
X_num = df_copy[numeric_features].fillna(0).values.astype(np.float32)

# ============================================
# STEP 4: Feature Engineering - TF-IDF on Clean Text
# ============================================
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df_copy["clean_text"].fillna(""))

# ============================================
# STEP 5: Combine Features
# ============================================
X = hstack([X_text, csr_matrix(X_num)]).tocsr()
y = df_copy["price"].values.astype(np.float32)

print("Feature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================
# STEP 6: Train/Validation Split and Model Training
# ============================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    valid_names=["val"],
    callbacks=[
        early_stopping(stopping_rounds=150),
        log_evaluation(period=100)
    ]
)


# ============================================
# STEP 7: Evaluation
# ============================================
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)


print("\n=== Validation Results ===")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("SMAPE:", smape(y_val, y_val_pred))

# ============================================
# STEP 8: Save Cleaned Data (Optional)
# ============================================
final_df = df_copy[["sample_id", "clean_text", "Value", "Unit", "price", "pack_count",
                    "total_weight_g", "total_volume_ml", "total_piece_ct"]]
final_df.to_csv("cleaned_train_data.csv", index=False)
print("\nCleaned data saved to cleaned_train_data.csv")
