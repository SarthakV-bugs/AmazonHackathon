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
print("\n" + "=" * 70)
print("MERGING ADDITIONAL FEATURES (IF NEEDED)")
print("=" * 70)

# Paths to your additional feature files
TRAIN_NEW_FEATURES = "path/to/additional_train_features.npz"  # UPDATE THIS
TEST_NEW_FEATURES = "path/to/additional_test_features.npz"  # UPDATE THIS

# Check if additional features exist and merge them
if os.path.exists(TRAIN_NEW_FEATURES):
    print("\nMerging train features...")
    existing = np.load("embeddings_train/combined_features_with_ids.npz")
    new = np.load(TRAIN_NEW_FEATURES)

    df_match = pd.merge(
        pd.DataFrame({'id': existing['sample_ids'], 'idx1': range(len(existing['sample_ids']))}),
        pd.DataFrame({'id': new['sample_ids'], 'idx2': range(len(new['sample_ids']))}),
        on='id', how='inner'
    )

    aligned_existing = existing['features'][df_match['idx1'].values]
    aligned_new = new['features'][df_match['idx2'].values]
    X_train_combined = np.concatenate([aligned_existing, aligned_new], axis=1)

    np.savez_compressed("embeddings_train/final_combined_features.npz",
                        sample_ids=df_match['id'].values,
                        features=X_train_combined)
    print(f"✓ Train features: {X_train_combined.shape}")

if os.path.exists(TEST_NEW_FEATURES):
    print("\nMerging test features...")
    existing = np.load("embeddings_test/combined_features_with_ids.npz")
    new = np.load(TEST_NEW_FEATURES)

    df_match = pd.merge(
        pd.DataFrame({'id': existing['sample_ids'], 'idx1': range(len(existing['sample_ids']))}),
        pd.DataFrame({'id': new['sample_ids'], 'idx2': range(len(new['sample_ids']))}),
        on='id', how='inner'
    )

    aligned_existing = existing['features'][df_match['idx1'].values]
    aligned_new = new['features'][df_match['idx2'].values]
    X_test_combined = np.concatenate([aligned_existing, aligned_new], axis=1)

    np.savez_compressed("embeddings_test/final_combined_features.npz",
                        sample_ids=df_match['id'].values,
                        features=X_test_combined)
    print(f"✓ Test features: {X_test_combined.shape}")
