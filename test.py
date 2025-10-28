import numpy as np
test_data = np.load("/home/ibab/PycharmProjects/AmazonHackathon/embeddings_test123_test/numeric_features_with_ids.npz")

# Check what arrays are stored inside
print("Keys in the .npz file:", test_data)

# Print the first few elements of each array
for key in test_data:
    print(f"\nArray name: {key}")
    print(test_data[key][:20])  # print first 5 elements like .head()
