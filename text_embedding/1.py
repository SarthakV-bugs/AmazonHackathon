# # import ssl
# # print(ssl.OPENSSL_VERSION)
# #
# #
# # import requests, urllib3, certifi
# # print("Requests:", requests.__version__)
# # print("urllib3:", urllib3.__version__)
# # print("Certifi CA path:", certifi.where())
# #
# #
# # from sentence_transformers import SentenceTransformer
# # model = SentenceTransformer("all-MiniLM-L6-v2")
#
# import pandas as pd
# df = pd.read_csv("/home/ibab/PycharmProjects/AmazonHackathon/student_resource/dataset/train.csv")
# print(df.shape)
# print(df.head())
# print(df.tail())
#
# print(df['catalog_content'].isna().sum())
# print(df['catalog_content'].str.strip().eq('').sum())
#
#
# print(df.duplicated().sum())
#
# import pandas as pd
#
# # Load CSV
# df = pd.read_csv("/home/ibab/PycharmProjects/AmazonHackathon/student_resource/dataset/train.csv")
#
# # Check for duplicates in a specific column, e.g., 'image_url' or 'sample_id'
# duplicate_links = df[df.duplicated(subset='image_link', keep=False)]  # keep=False shows all duplicates
#
# print("Number of duplicate links:", duplicate_links.shape[0])
# print("Duplicate rows:")
# print(duplicate_links)
#
# num_unique_duplicates = df['image_url'][df['image_url'].duplicated()].nunique()
# print("Number of unique duplicate links:", num_unique_duplicates)

import numpy as np
data = np.load("../text_embedding/embeddings/sbert_embeddings.npy")


