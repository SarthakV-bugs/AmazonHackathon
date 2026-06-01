import os
import re
import gc
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
tqdm.pandas()

# -------------------------
# Text Utilities
# -------------------------
def clean_extract(text):
    text = str(text)
    v = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(v.group(1)) if v else 0
    u = re.search(r"Unit:\s*([A-Za-z ]+)", text)
    unit = u.group(1).strip() if u else None
    text = re.sub(r"Value:\s*[\d\.]+", "", text)
    text = re.sub(r"Unit:\s*[A-Za-z ]+", "", text)
    text = re.sub(r"Item Name:\s*", "", text)
    text = re.sub(r"Bullet Point\s*\d*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r\t\xa0]+", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return pd.Series([text, value, unit])

PACK_PATTERNS = [
    r"(?:pack|case)\s*of\s*(\d+)", r"(\d+)\s*-\s*pack", r"(\d+)\s*count",
    r"\((\d+)\s*count\)", r"\((\d+)\s*ct\)", r"(\d+)\s*ct\b",
    r"\(\s*pack\s*of\s*(\d+)\s*\)", r"\b(\d+)\s*each\b", r"\b(\d+)\s*ea\b",
]
WEIGHT_UNITS = {"oz":28.35,"ounce":28.35,"lb":453.59,"pound":453.59,"g":1.0,"gram":1.0,"kg":1000.0}
VOLUME_UNITS = {"fl oz":29.57,"ml":1.0,"l":1000.0}
COUNT_TOKENS = {"count","ct","each","ea"}

def extract_pack_count(text):
    s = str(text).lower()
    for p in PACK_PATTERNS:
        m = re.search(p, s)
        if m:
            try: return int(m.group(1))
            except: pass
    return 1

def normalize_quantity(value, unit):
    if unit is None or (isinstance(unit,float) and np.isnan(unit)):
        return 0.0,0.0,0.0
    u = unit.strip().lower().replace("ounces","ounce").replace("lbs","lb").replace("grams","gram")
    if u in WEIGHT_UNITS: return value*WEIGHT_UNITS[u],0.0,0.0
    if u in VOLUME_UNITS: return 0.0,value*VOLUME_UNITS[u],0.0
    if u in COUNT_TOKENS: return 0.0,0.0,value
    return 0.0,0.0,0.0

# -------------------------
# Text Feature Extractor
# -------------------------
class TextFeatureExtractor:
    def __init__(self, sbert_model_path="./all-MiniLM-L6-v2_local",
                 tfidf_max_features=15000, svd_n_components=256):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        self.sbert_model = SentenceTransformer(sbert_model_path)
        self.tfidf_vec = None
        self.svd_model = None
        self.tfidf_max_features = tfidf_max_features
        self.svd_n_components = svd_n_components

    def _get_text_column(self, df):
        for c in df.columns:
            if any(k in c.lower() for k in ["catalog", "description", "text", "content", "title"]):
                return c
        raise ValueError("No suitable text column found!")

    def process(self, df, fit=True, batch_size=512):
        df = df.copy()
        text_col = self._get_text_column(df)
        df[["clean_text","Value","Unit"]] = df[text_col].progress_apply(clean_extract)
        df["pack_count"] = df["clean_text"].progress_apply(extract_pack_count)
        qtys = [normalize_quantity(r["Value"],r["Unit"]) for _,r in df.iterrows()]
        df["total_weight_g"],df["total_volume_ml"],df["total_piece_ct"]=zip(*qtys)
        num = df[["pack_count","total_weight_g","total_volume_ml","total_piece_ct"]].fillna(0).astype(np.float32).values
        num_log = np.log1p(num)

        texts = df["clean_text"].fillna("").tolist()
        embs=[]
        for i in range(0,len(texts),batch_size):
            batch=texts[i:i+batch_size]
            embs.append(self.sbert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True))
        sbert_feats = np.vstack(embs)

        if fit:
            self.tfidf_vec = TfidfVectorizer(max_features=self.tfidf_max_features, ngram_range=(1,2))
            X_sparse = self.tfidf_vec.fit_transform(df["clean_text"])
            self.svd_model = TruncatedSVD(n_components=self.svd_n_components, random_state=42)
            X_text = self.svd_model.fit_transform(X_sparse)
        else:
            if self.tfidf_vec is None or self.svd_model is None:
                raise RuntimeError("TF-IDF or SVD not fitted yet.")
            X_sparse = self.tfidf_vec.transform(df["clean_text"])
            X_text = self.svd_model.transform(X_sparse)

        return df["sample_id"].astype(str).values, sbert_feats, X_text, num_log

# -------------------------
# Image Feature Extractor
# -------------------------
class ImageFeatureExtractor:
    def __init__(self, device=None, weights_path="./efficientnet_b0_imagenet1k_v1.pth"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_b0(weights=None).to(self.device)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"EfficientNet weights not found: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.classifier = nn.Identity()
        self.model.eval()

        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    # ✅ FIXED preprocess_image to handle missing files
    def preprocess_image(self, img_path):
        if img_path is None or not os.path.exists(img_path):
            # Return dummy image (all zeros)
            return torch.zeros(3, 224, 224)
        try:
            img = Image.open(img_path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"[WARN] Failed to open/process image {img_path}, using zeros.")
            return torch.zeros(3, 224, 224)

    # ✅ FIXED extract_features with caching duplicates
    def extract_features(self, paths, batch_size=64, n_workers=8):
        cache = {}
        all_feats = []

        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            unique_batch = list(dict.fromkeys(p for p in batch if p is not None))
            new_imgs = [p for p in unique_batch if p not in cache]

            if new_imgs:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    tensors = list(ex.map(self.preprocess_image, new_imgs))
                tb = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    feats = self.model(tb).cpu().numpy()
                for p, f in zip(new_imgs, feats):
                    cache[p] = f

            batch_feats = [cache.get(p, np.zeros(1280)) for p in batch]
            all_feats.append(np.stack(batch_feats))

        return np.vstack(all_feats)

    # ✅ FIXED process
    def process(self, df, image_dir, batch_size=128):
        if not os.path.exists(image_dir):
            alt_dir = os.path.dirname(image_dir.rstrip("/"))
            if os.path.exists(alt_dir):
                print(f"[WARN] Using fallback image dir: {alt_dir}")
                image_dir = alt_dir
            else:
                raise FileNotFoundError(f"Image directory not found: {image_dir}")

        ids, paths = [], []
        for sid, link in zip(df["sample_id"], df.get("image_link", [])):
            ids.append(str(sid))
            if pd.isna(link) or not isinstance(link, str):
                paths.append(None)
            else:
                fname = os.path.basename(link.split("?")[0])
                paths.append(os.path.join(image_dir, fname))

        feats = self.extract_features(paths, batch_size=batch_size)
        return np.array(ids), feats

# -------------------------
# Combined Processor
# -------------------------
class MultimodalFeatureProcessor:
    def __init__(self, text_extractor, image_extractor):
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor

    def process_split(self, df, image_dir, fit_text=True):
        if df.empty:
            raise ValueError("Input dataframe is empty!")
        print(f"\n🔹 Processing split | fit_text={fit_text}")
        print(f"Input samples: {len(df)}")

        ids_txt, sbert_f, tfidf_f, num_f = self.text_extractor.process(df, fit=fit_text)
        print(f"  SBERT:  {sbert_f.shape}")
        print(f"  TF-IDF: {tfidf_f.shape}")
        print(f"  NUM:    {num_f.shape}")

        ids_img, img_f = self.image_extractor.process(df, image_dir)
        print(f"  IMG:    {img_f.shape}")

        df_txt = pd.DataFrame({"sample_id": ids_txt, "txt_idx": np.arange(len(ids_txt))})
        df_img = pd.DataFrame({"sample_id": ids_img, "img_idx": np.arange(len(ids_img))})
        common = pd.merge(df_txt, df_img, on="sample_id", how="inner")

        ids_common = common["sample_id"].values
        sbert_f_aligned = np.take(sbert_f, common["txt_idx"], axis=0)
        tfidf_f_aligned = np.take(tfidf_f, common["txt_idx"], axis=0)
        num_f_aligned = np.take(num_f, common["txt_idx"], axis=0)
        img_f_aligned = np.take(img_f, common["img_idx"], axis=0)

        combined = np.hstack([tfidf_f_aligned, sbert_f_aligned, num_f_aligned, img_f_aligned])
        print(f"✅ Combined multimodal shape: {combined.shape}")

        return ids_common, combined

# -------------------------
# Main Pipeline
# -------------------------
def main(dataset_dir="../dataset"):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    text_extractor = TextFeatureExtractor()
    image_extractor = ImageFeatureExtractor()
    processor = MultimodalFeatureProcessor(text_extractor, image_extractor)

    for split in ["train","test"]:
        csv_path = os.path.join(dataset_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        img_dir = os.path.join(dataset_dir, f"{split}_images/")

        ids, features = processor.process_split(df, img_dir, fit_text=(split=="train"))

        np.savez_compressed(
            os.path.join(dataset_dir, f"{split}_features.npz"),
            sample_ids=ids,
            features=features
        )
        print(f"💾 Saved {split}_features.npz with shape {features.shape}")

        gc.collect()
        torch.cuda.empty_cache()

if __name__=="__main__":
    main()
