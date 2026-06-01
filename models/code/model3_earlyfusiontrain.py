import torch.nn.functional as F

# Encoder for each modality
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

# Fusion model with encoders + prediction head
class LearnedFusionModel(nn.Module):
    def __init__(self, dims_text_tfidf, dims_text_sbert, dims_num, dims_img, fused_dim=128):
        super().__init__()
        self.encoder_tfidf = ModalityEncoder(dims_text_tfidf)
        self.encoder_sbert = ModalityEncoder(dims_text_sbert)
        self.encoder_num = ModalityEncoder(dims_num)
        self.encoder_img = ModalityEncoder(dims_img)

        fusion_input_dim = (dims_text_tfidf // 2) + (dims_text_sbert // 2) + (dims_num // 2) + (dims_img // 2)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

        self.pred_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Price regression output
        )

    def forward(self, tfidf, sbert, num, img, return_fusion_vector=False):
        e_tfidf = self.encoder_tfidf(tfidf)
        e_sbert = self.encoder_sbert(sbert)
        e_num = self.encoder_num(num)
        e_img = self.encoder_img(img)
        fused = torch.cat([e_tfidf, e_sbert, e_num, e_img], dim=1)
        fused_vector = self.fusion_net(fused)
        if return_fusion_vector:
            return fused_vector
        pred = self.pred_head(fused_vector)
        return pred.squeeze(1), fused_vector

# Custom SMAPE loss (optional)
def smape_loss(preds, targets, epsilon=1e-6):
    denominator = (torch.abs(targets) + torch.abs(preds)).clamp(min=epsilon)
    diff = torch.abs(preds - targets)
    loss = 2 * diff / denominator
    return loss.mean()

# Training loop
def train_model(model, train_feats, train_prices, val_feats, val_prices,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                epochs=10, batch_size=64, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = smape_loss  # or torch.nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        *(torch.tensor(f).float() for f in train_feats),
        torch.tensor(train_prices).float()
    )
    val_dataset = torch.utils.data.TensorDataset(
        *(torch.tensor(f).float() for f in val_feats),
        torch.tensor(val_prices).float()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            *inputs, targets = batch
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            optimizer.zero_grad()
            preds, _ = model(*inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                *inputs, targets = batch
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                preds, _ = model(*inputs)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {avg_train:.4f} | Val loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_fusion_model.pth")
            print("-> Saved best model")

    return model

# Modified MultimodalFeatureProcessor.process_split to return separate features
class MultimodalFeatureProcessor:
    # your existing __init__ here...

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

        print(f"✅ Aligned multimodal features shapes:")
        print(f"  TF-IDF: {tfidf_f_aligned.shape}")
        print(f"  SBERT:  {sbert_f_aligned.shape}")
        print(f"  NUM:    {num_f_aligned.shape}")
        print(f"  IMG:    {img_f_aligned.shape}")

        return ids_common, tfidf_f_aligned, sbert_f_aligned, num_f_aligned, img_f_aligned

# Your main function: loading data, training fusion + predictor head jointly
def main(dataset_dir="../dataset"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    text_extractor = TextFeatureExtractor()
    image_extractor = ImageFeatureExtractor()
    processor = MultimodalFeatureProcessor(text_extractor, image_extractor)

    # Load train split, extract features
    train_csv = os.path.join(dataset_dir, "train.csv")
    df_train = pd.read_csv(train_csv)
    train_img_dir = os.path.join(dataset_dir, "train_images/")
    train_ids, train_tfidf, train_sbert, train_num, train_img = processor.process_split(df_train, train_img_dir, fit_text=True)

    # Extract target prices (replace 'price' with your actual column name)
    train_prices = df_train.set_index("sample_id").loc[train_ids]["price"].values.astype(np.float32)

    # Split train/val (e.g., 90/10)
    val_split = 0.1
    n_val = int(len(train_ids) * val_split)

    train_feats = (train_tfidf[:-n_val], train_sbert[:-n_val], train_num[:-n_val], train_img[:-n_val])
    val_feats = (train_tfidf[-n_val:], train_sbert[-n_val:], train_num[-n_val:], train_img[-n_val:])
    train_prices_split = train_prices[:-n_val]
    val_prices_split = train_prices[-n_val:]

    dims = (train_tfidf.shape[1], train_sbert.shape[1], train_num.shape[1], train_img.shape[1])

    print(f"\nStarting training fusion model with dims: {dims}")
    model = LearnedFusionModel(*dims).to(device)

    # Train the fusion + prediction head model end-to-end
    model = train_model(model, train_feats, train_prices_split, val_feats, val_prices_split, device=device)

    # After training, save learned fusion embeddings for train set
    model.eval()
    with torch.no_grad():
        train_tensors = [torch.tensor(f).float().to(device) for f in (train_tfidf, train_sbert, train_num, train_img)]
        fused_embeddings = model(*train_tensors, return_fusion_vector=True).cpu().numpy()

    np.savez_compressed(os.path.join(dataset_dir, "train_learned_fusion_embeddings.npz"), sample_ids=train_ids, embeddings=fused_embeddings)
    print(f"Saved learned fusion embeddings for train set: {fused_embeddings.shape}")

    # Similarly process test set (load features, get fusion embeddings)
    test_csv = os.path.join(dataset_dir, "test.csv")
    df_test = pd.read_csv(test_csv)
    test_img_dir = os.path.join(dataset_dir, "test_images/")
    test_ids, test_tfidf, test_sbert, test_num, test_img = processor.process_split(df_test, test_img_dir, fit_text=False)

    with torch.no_grad():
        test_tensors = [torch.tensor(f).float().to(device) for f in (test_tfidf, test_sbert, test_num, test_img)]
        test_fused_embeddings = model(*test_tensors, return_fusion_vector=True).cpu().numpy()

    np.savez_compressed(os.path.join(dataset_dir, "test_learned_fusion_embeddings.npz"), sample_ids=test_ids, embeddings=test_fused_embeddings)
    print(f"Saved learned fusion embeddings for test set: {test_fused_embeddings.shape}")

    gc.collect()
    torch.cuda.empty_cache()
