import pandas as pd

# Load your original CSV
df = pd.read_csv("multimodal_predictions.csv")  # Change filename if different

# Create ensemble CSV (sample_id + ensemble predictions)
df_ensemble = df[["sample_id", "ensemble"]].copy()
df_ensemble.columns = ["sample_id", "predicted_price"]  # Rename for submission format
df_ensemble.to_csv("ensemble_predictions.csv", index=False)

# Create CNN CSV (sample_id + cnn predictions)
df_cnn = df[["sample_id", "cnn"]].copy()
df_cnn.columns = ["sample_id", "predicted_price"]  # Rename for submission format
df_cnn.to_csv("cnn_predictions.csv", index=False)

print(f"✓ Created ensemble_predictions.csv: {len(df_ensemble)} rows")
print(f"✓ Created cnn_predictions.csv: {len(df_cnn)} rows")
