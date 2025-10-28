import numpy as np
import os

# Define the folder where embeddings are saved
embedding_folder = "embeddings_test123"

# Load all three npz files
print("Loading SBERT embeddings...")
sbert_data = np.load(os.path.join(embedding_folder, "sbert_embeddings_with_ids.npz"))
sbert_ids = sbert_data["sample_ids"]
sbert_embeddings = sbert_data["embeddings"]

print("Loading TF-IDF dense features...")
tfidf_data = np.load(os.path.join(embedding_folder, "tfidf_dense_features_with_ids.npz"))
tfidf_ids = tfidf_data["sample_ids"]
tfidf_dense = tfidf_data["tfidf_dense"]

print("Loading numeric features...")
numeric_data = np.load(os.path.join(embedding_folder, "numeric_features_with_ids.npz"))
numeric_ids = numeric_data["sample_ids"]
numeric_features = numeric_data["numeric_features"]

# Verify that all sample_ids match
assert np.array_equal(sbert_ids, tfidf_ids), "SBERT and TF-IDF sample_ids don't match!"
assert np.array_equal(sbert_ids, numeric_ids), "SBERT and numeric sample_ids don't match!"
print(f"✓ All sample_ids match! Total samples: {len(sbert_ids)}")

# Concatenate features horizontally (along axis=1)
combined_features = np.concatenate([tfidf_dense, sbert_embeddings, numeric_features], axis=1)
print(f"Combined features shape: {combined_features.shape}")
print(f"  - TF-IDF dense: {tfidf_dense.shape[1]} features")
print(f"  - SBERT embeddings: {sbert_embeddings.shape[1]} features")
print(f"  - Numeric features: {numeric_features.shape[1]} features")
print(f"  - Total: {combined_features.shape[1]} features")

# Save the combined file
output_path = os.path.join(embedding_folder, "combined_features_with_ids.npz")
np.savez_compressed(output_path,
                    sample_ids=sbert_ids,
                    features=combined_features)

print(f"✓ Saved combined features to: {output_path}")

# Verify by loading and checking
verify = np.load(output_path)
print(f"\nVerification:")
print(f"  - sample_ids shape: {verify['sample_ids'].shape}")
print(f"  - features shape: {verify['features'].shape}")
