import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
# Text cleaning and extraction functions
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


# -------------------------
# Function to process dataset (train or test)
def process_dataset(df, dataset_type, sbert_model, tfidf_vectorizer=None, svd_model=None):
    """
    Process dataset and extract features

    Args:
        df: Input dataframe
        dataset_type: 'train' or 'test'
        sbert_model: Pre-loaded SBERT model
        tfidf_vectorizer: Fitted TF-IDF vectorizer (None for train, fitted for test)
        svd_model: Fitted SVD model (None for train, fitted for test)

    Returns:
        Dictionary containing features and models
    """
    print(f"\n{'=' * 50}")
    print(f"Processing {dataset_type.upper()} dataset...")
    print(f"{'=' * 50}")

    df_copy = df.copy()

    # Text cleaning and extraction
    print("Cleaning and extracting structured info...")
    df_copy[["clean_text", "Value", "Unit"]] = df_copy["catalog_content"].progress_apply(clean_extract)

    # Extract pack count
    print("Extracting pack count...")
    df_copy["pack_count"] = df_copy["clean_text"].progress_apply(extract_pack_count)

    # Normalize quantities
    print("Normalizing quantities...")
    results = [normalize_quantity(row["Value"], row["Unit"])
               for _, row in tqdm(df_copy.iterrows(), total=len(df_copy))]
    df_copy["total_weight_g"], df_copy["total_volume_ml"], df_copy["total_piece_ct"] = zip(*results)

    # SBERT embeddings
    print("Generating SBERT embeddings...")
    BATCH = 512
    texts = df_copy["clean_text"].fillna("").tolist()
    sample_ids = df_copy["sample_id"].values
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH), desc="Encoding SBERT batches"):
        batch = texts[i:i + BATCH]
        emb = sbert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"SBERT embeddings shape: {embeddings.shape}")

    # Numeric features
    numeric_features = ["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]
    X_num = df_copy[numeric_features].fillna(0).astype(np.float32).values
    X_num_log = np.log1p(X_num)

    # TF-IDF features with TruncatedSVD
    print("Building TF-IDF features...")
    if tfidf_vectorizer is None:  # Training phase
        tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        X_text_sparse = tfidf_vectorizer.fit_transform(
            tqdm(df_copy["clean_text"].fillna(""), desc="TF-IDF fitting")
        )
    else:  # Test phase - use fitted vectorizer
        X_text_sparse = tfidf_vectorizer.transform(
            tqdm(df_copy["clean_text"].fillna(""), desc="TF-IDF transforming")
        )

    # Convert TF-IDF to dense using TruncatedSVD
    n_components = 256
    if svd_model is None:  # Training phase
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        X_text = svd_model.fit_transform(X_text_sparse)
    else:  # Test phase - use fitted SVD
        X_text = svd_model.transform(X_text_sparse)

    print(f"TF-IDF dense features shape after TruncatedSVD: {X_text.shape}")

    return {
        'sample_ids': sample_ids,
        'sbert_embeddings': embeddings,
        'tfidf_dense': X_text,
        'numeric_features': X_num_log,
        'tfidf_vectorizer': tfidf_vectorizer,
        'svd_model': svd_model,
        'df_processed': df_copy
    }


# -------------------------
# Function to save embeddings to npz files
def save_embeddings(output_folder, sample_ids, sbert_embeddings, tfidf_dense, numeric_features):
    """Save embeddings to separate npz files"""
    os.makedirs(output_folder, exist_ok=True)

    # Save SBERT embeddings
    np.savez_compressed(
        os.path.join(output_folder, "sbert_embeddings_with_ids.npz"),
        sample_ids=sample_ids,
        embeddings=sbert_embeddings
    )

    # Save TF-IDF dense features
    np.savez_compressed(
        os.path.join(output_folder, "tfidf_dense_features_with_ids.npz"),
        sample_ids=sample_ids,
        tfidf_dense=tfidf_dense
    )

    # Save numeric features
    np.savez_compressed(
        os.path.join(output_folder, "numeric_features_with_ids.npz"),
        sample_ids=sample_ids,
        numeric_features=numeric_features
    )

    print(f"Saved separate embeddings to: {output_folder}")


# -------------------------
# Function to combine embeddings into single npz
def combine_embeddings(embedding_folder):
    """Combine three separate npz files into one"""
    print(f"\nCombining embeddings from {embedding_folder}...")

    # Load all three npz files
    sbert_data = np.load(os.path.join(embedding_folder, "sbert_embeddings_with_ids.npz"))
    sbert_ids = sbert_data["sample_ids"]
    sbert_embeddings = sbert_data["embeddings"]

    tfidf_data = np.load(os.path.join(embedding_folder, "tfidf_dense_features_with_ids.npz"))
    tfidf_ids = tfidf_data["sample_ids"]
    tfidf_dense = tfidf_data["tfidf_dense"]

    numeric_data = np.load(os.path.join(embedding_folder, "numeric_features_with_ids.npz"))
    numeric_ids = numeric_data["sample_ids"]
    numeric_features = numeric_data["numeric_features"]

    # Verify that all sample_ids match
    assert np.array_equal(sbert_ids, tfidf_ids), "SBERT and TF-IDF sample_ids don't match!"
    assert np.array_equal(sbert_ids, numeric_ids), "SBERT and numeric sample_ids don't match!"
    print(f"✓ All sample_ids match! Total samples: {len(sbert_ids)}")

    # Concatenate features horizontally
    combined_features = np.concatenate([tfidf_dense, sbert_embeddings, numeric_features], axis=1)
    print(f"Combined features shape: {combined_features.shape}")
    print(f"  - TF-IDF dense: {tfidf_dense.shape[1]} features")
    print(f"  - SBERT embeddings: {sbert_embeddings.shape[1]} features")
    print(f"  - Numeric features: {numeric_features.shape[1]} features")
    print(f"  - Total: {combined_features.shape[1]} features")

    # Save combined file
    output_path = os.path.join(embedding_folder, "combined_features_with_ids.npz")
    np.savez_compressed(output_path,
                        sample_ids=sbert_ids,
                        features=combined_features)

    print(f"✓ Saved combined features to: {output_path}")

    return combined_features, sbert_ids


# -------------------------
# MAIN PIPELINE
# -------------------------

print("=" * 70)
print("STARTING UNIFIED TRAIN-TEST PIPELINE")
print("=" * 70)

# Load SBERT model once
print("\nLoading SBERT model...")
sbert_model_name = "all-MiniLM-L6-v2"
sbert = SentenceTransformer(sbert_model_name)

# -------------------------
# PROCESS TRAINING DATA
# -------------------------
print("\n" + "=" * 70)
print("PHASE 1: TRAINING DATA PROCESSING")
print("=" * 70)

df_train = pd.read_csv("/home/ibab/PycharmProjects/AmazonHackathon/student_resource/dataset/train.csv")
train_results = process_dataset(df_train, "train", sbert)

# Save train embeddings
train_folder = "embeddings_train"
save_embeddings(
    train_folder,
    train_results['sample_ids'],
    train_results['sbert_embeddings'],
    train_results['tfidf_dense'],
    train_results['numeric_features']
)

# Combine train embeddings
X_train_combined, train_ids = combine_embeddings(train_folder)

# -------------------------
# PROCESS TEST DATA
# -------------------------
print("\n" + "=" * 70)
print("PHASE 2: TEST DATA PROCESSING")
print("=" * 70)

df_test = pd.read_csv("/home/ibab/PycharmProjects/AmazonHackathon/student_resource/dataset/test.csv")
test_results = process_dataset(
    df_test,
    "test",
    sbert,
    tfidf_vectorizer=train_results['tfidf_vectorizer'],
    svd_model=train_results['svd_model']
)

# Save test embeddings
test_folder = "embeddings_test"
save_embeddings(
    test_folder,
    test_results['sample_ids'],
    test_results['sbert_embeddings'],
    test_results['tfidf_dense'],
    test_results['numeric_features']
)

# Combine test embeddings
X_test_combined, test_ids = combine_embeddings(test_folder)

# -------------------------
# TRAIN LIGHTGBM MODEL
# -------------------------
print("\n" + "=" * 70)
print("PHASE 3: MODEL TRAINING")
print("=" * 70)

# Convert to sparse matrices for LightGBM
X_train_sparse = csr_matrix(X_train_combined)
y_train = train_results['df_processed']["price"].values.astype(np.float32)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sparse, y_train, test_size=0.2, random_state=42
)


# Define SMAPE loss
def smape_loss_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    loss = 100.0 * np.mean(np.abs(y_pred - y_true) / denom)
    return "SMAPE", loss, False


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)


# LightGBM parameters
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
lgb_train = lgb.Dataset(X_tr, label=y_tr)
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
# EVALUATE MODEL
# -------------------------
print("\n" + "=" * 70)
print("PHASE 4: MODEL EVALUATION")
print("=" * 70)

y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
print(f"Validation MAE: {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"Validation SMAPE: {smape(y_val, y_val_pred):.4f}")

# -------------------------
# SAVE MODEL AND ARTIFACTS
# -------------------------
print("\n" + "=" * 70)
print("PHASE 5: SAVING MODEL AND ARTIFACTS")
print("=" * 70)

print("Saving model and artifacts...")
model.save_model("lgb_sbert_model.txt")
joblib.dump(train_results['tfidf_vectorizer'], "tfidf.pkl")
joblib.dump(train_results['svd_model'], "svd.pkl")
numeric_features = ["pack_count", "total_weight_g", "total_volume_ml", "total_piece_ct"]
np.save("numeric_cols.npy", np.array(numeric_features))

print("\n" + "=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - Train samples: {len(train_ids)}")
print(f"  - Test samples: {len(test_ids)}")
print(f"  - Feature dimensions: {X_train_combined.shape[1]}")
print(f"  - Train embeddings folder: {train_folder}/")
print(f"  - Test embeddings folder: {test_folder}/")
print(f"  - Combined train file: {train_folder}/combined_features_with_ids.npz")
print(f"  - Combined test file: {test_folder}/combined_features_with_ids.npz")
