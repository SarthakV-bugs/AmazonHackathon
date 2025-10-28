import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
tqdm.pandas()

# -------------------------
# Proxy setup (IBAB network)
os.environ["HTTP_PROXY"] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://245hsbd009%40ibab.ac.in:verma1903@proxy.ibab.ac.in:3128"

# -------------------------
# Load data
print("Loading dataset...")
df = pd.read_csv("/home/ibab/PycharmProjects/AmazonHackathon/student_resource/dataset/train.csv")
df_copy = df.copy()

# -------------------------
# Text cleaning and extraction
def clean_extract(text):
    text = str(text)
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(value_match.group(1)) if value_match else 0
    unit_match = re.search(r"Unit:\s*([A-Za-z ]+)", text)
    unit = unit_match.group(1).strip() if unit_match else None

    text = re.sub(r"Value:\s*[\d\.]+", "", text)
    text = re.sub(r"Unit:\s*[A-Za-z ]+", "", text)
    text = re.sub(r"Item Name:\s*", "", text)
    text = re.sub(r"Bullet Point\s*\d*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r\t\xa0]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    return pd.Series([text, value, unit])

print("Cleaning and extracting structured info...")
df_copy[["clean_text", "Value", "Unit"]] = df_copy["catalog_content"].progress_apply(clean_extract)

# -------------------------
# Extract pack counts
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

print("Extracting pack count...")
df_copy["pack_count"] = df_copy["clean_text"].progress_apply(extract_pack_count)

# -------------------------
# Normalize quantities
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

print("Normalizing quantities...")
results = [normalize_quantity(row["Value"], row["Unit"]) for _, row in tqdm(df_copy.iterrows(), total=len(df_copy))]
df_copy["total_weight_g"], df_copy["total_volume_ml"], df_copy["total_piece_ct"] = zip(*results)

# -------------------------
# -------------------------
# SBERT embeddings
print("Generating SBERT embeddings...")
sbert_model_name = "all-mpnet-base-v2"
sbert = SentenceTransformer(sbert_model_name)

BATCH = 512
texts = df_copy["clean_text"].fillna("").tolist()
sample_ids = df_copy["sample_id"].values  # keep sample IDs
embeddings = []

for i in tqdm(range(0, len(texts), BATCH), desc="Encoding SBERT batches"):
    batch = texts[i:i + BATCH]
    emb = sbert.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)  # shape: (num_rows, embedding_dim)
print("SBERT embeddings shape:", embeddings.shape)

# -------------------------
# Save SBERT embeddings and sample_ids together in a compressed .npz file
os.makedirs("embeddings_test123", exist_ok=True)
np.savez_compressed("embeddings_test123/sbert_embeddings_with_ids.npz",
                    sample_ids=sample_ids,
                    embeddings=embeddings)

print(f"Saved SBERT embeddings and sample_ids with shapes: {embeddings.shape}, {sample_ids.shape}")

# -------------------------
# Numeric features
numeric_features = ["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]
X_num = df_copy[numeric_features].fillna(0).astype(np.float32).values
X_num_log = np.log1p(X_num)

# Save numeric features with sample_id
np.savez_compressed("embeddings_test123/numeric_features_with_ids.npz",
                    sample_ids=df_copy["sample_id"].values,
                    numeric_features=X_num_log)
print(f"Saved numeric features with shape: {X_num_log.shape}")

# -------------------------
# TF-IDF features with TruncatedSVD (dense)
from sklearn.decomposition import TruncatedSVD

print("Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_text_sparse = tfidf.fit_transform(tqdm(df_copy["clean_text"].fillna(""), desc="TF-IDF fitting"))

# Convert TF-IDF to dense using TruncatedSVD
n_components = 256
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_text = svd.fit_transform(X_text_sparse)
print(f"TF-IDF dense features shape after TruncatedSVD: {X_text.shape}")

# Save dense TF-IDF features with sample_id
np.savez_compressed("embeddings_test123/tfidf_dense_features_with_ids.npz",
                    sample_ids=df_copy["sample_id"].values,
                    tfidf_dense=X_text)

# -------------------------
# Combine features
X_sbert_sparse = csr_matrix(embeddings)
X_num_sparse = csr_matrix(X_num_log)
X = hstack([X_text, X_sbert_sparse, X_num_sparse]).tocsr()
y = df_copy["price"].values.astype(np.float32)
print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

# -------------------------
# Train LightGBM
def smape_loss_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    loss = 100.0 * np.mean(np.abs(y_pred - y_true) / denom)
    return "SMAPE", loss, False

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1
}

print("Training LightGBM model...")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_val],
    valid_names=["val"],
    feval=smape_loss_lgb,
    callbacks=[early_stopping(stopping_rounds=150), log_evaluation(period=100)]
)

# -------------------------
# Evaluate
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("SMAPE:", smape(y_val, y_val_pred))

# -------------------------
# Save artifacts
print("Saving model and artifacts...")
model.save_model("lgb_sbert_model.txt")
joblib.dump(tfidf, "tfidf.pkl")
np.save("numeric_cols.npy", np.array(numeric_features))
print("Pipeline completed successfully!")
