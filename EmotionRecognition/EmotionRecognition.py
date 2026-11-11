
"""
VREED pipeline: eye-tracking, multimodal + ViT, transferencia a datos mineros.
Requiere: pandas, numpy, scikit-learn, torch, torchvision, opencv-python
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import datasets, transforms, models

# -----------------------------
# 1) CARGA CSV EYE-TRACKING
# -----------------------------

LABEL_4F = "label_4f"       # emoción 4 factores
LABEL_AROUSAL = "arousal"   # discreto o binarizado
LABEL_VALENCE = "valence"   # discreto o binarizado

META_COLS = ["subject_id", "trial_id", "timestamp"]


def load_eyetracking_csv(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def get_feature_cols(df: pd.DataFrame):
    return [
        c
        for c in df.columns
        if c not in META_COLS + [LABEL_4F, LABEL_AROUSAL, LABEL_VALENCE]
    ]


# -----------------------------
# 2) BASELINE CLASIFICACIÓN TABULAR (5-FOLD)
# -----------------------------

def run_tabular_kfold(X, y, n_splits=5, model=None, task_name="4F"):
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        f1 = f1_score(y_te, y_pred, average="macro")
        f1_scores.append(f1)
        print(f"[{task_name}] Fold {fold} F1-macro: {f1:.3f}")

    print(f"[{task_name}] F1-macro medio: {np.mean(f1_scores):.3f}")
    return f1_scores


# -----------------------------
# 3) DATASET SECUENCIAL PARA TRANSFORMER (UNIMODAL)
# -----------------------------

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # lista de np.array [T_i, D]
        self.labels = labels        # np.array [N]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx].astype(np.float32)  # [T, D]
        y = self.labels[idx]
        return x, y


def build_sequences(df: pd.DataFrame, feature_cols, label_col):
    seqs = []
    labels = []

    grouped = df.groupby(["subject_id", "trial_id"], sort=True)
    for (_, _), g in grouped:
        x = g[feature_cols].values
        y = g[label_col].iloc[0]  # etiqueta del trial
        seqs.append(x)
        labels.append(y)

    return seqs, np.array(labels)


def collate_padded(batch):
    # batch: list of (np.array[T,D], label)
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].shape[1]

    x_pad = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)

    for i, s in enumerate(seqs):
        L = s.shape[0]
        x_pad[i, :L] = torch.from_numpy(s)
        mask[i, :L] = True

    y = torch.tensor(labels, dtype=torch.long)
    return x_pad, mask, y


# -----------------------------
# 4) TRANSFORMER TEMPORAL UNIMODAL (EYE-TRACKING)
# -----------------------------

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        """
        x: [B, T, D_in]
        mask: [B, T] True donde hay datos válidos
        """
        x = self.proj(x)  # [B,T,d_model]
        key_padding_mask = ~mask  # Transformer: True = padding
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # average pooling sobre time usando mask
        mask_f = mask.unsqueeze(-1).float()
        h_sum = (h * mask_f).sum(dim=1)
        lengths = mask_f.sum(dim=1).clamp(min=1.0)
        h_mean = h_sum / lengths

        logits = self.cls_head(h_mean)
        return logits


def train_transformer_5fold(seqs, labels, num_classes, n_splits=5, epochs=10,
                            batch_size=16, lr=1e-3, device="cuda"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = np.asarray(labels)
    f1_scores = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        train_ds = SequenceDataset([seqs[i] for i in tr_idx], labels[tr_idx])
        test_ds = SequenceDataset([seqs[i] for i in te_idx], labels[te_idx])

        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_padded
        )
        test_dl = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_padded
        )

        input_dim = seqs[0].shape[1]
        model = TemporalTransformer(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Entrenamiento
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, mb, yb in train_dl:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb, mb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            print(f"Epoch {epoch}: loss = {running_loss/len(train_ds):.4f}")

        # Evaluación
        model.eval()
        all_y, all_pred = [], []
        with torch.no_grad():
            for xb, mb, yb in test_dl:
                xb, mb = xb.to(device), mb.to(device)
                logits = model(xb, mb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_pred.extend(preds)
                all_y.extend(yb.numpy())

        f1 = f1_score(all_y, all_pred, average="macro")
        print(classification_report(all_y, all_pred))
        print(f"Fold {fold} F1-macro Transformer: {f1:.3f}")
        f1_scores.append(f1)

    print(f"\nF1-macro medio Transformer: {np.mean(f1_scores):.3f}")
    return f1_scores


# -----------------------------
# 5) MULTIMODAL: EYE-TRACKING + FISIO (ECG/GSR)
# -----------------------------

class MultiModalSequenceDataset(Dataset):
    def __init__(self, eye_seqs, physio_seqs, labels):
        assert len(eye_seqs) == len(physio_seqs) == len(labels)
        self.eye = eye_seqs
        self.phys = physio_seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_eye = self.eye[idx].astype(np.float32)
        x_phy = self.phys[idx].astype(np.float32)
        y = self.labels[idx]
        return x_eye, x_phy, y


def collate_multimodal(batch):
    eye_seqs, phy_seqs, labels = zip(*batch)
    # padding independiente por modalidad
    def pad_mod(seqs):
        lengths = [s.shape[0] for s in seqs]
        max_len = max(lengths)
        dim = seqs[0].shape[1]
        x_pad = torch.zeros(len(seqs), max_len, dim, dtype=torch.float32)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = s.shape[0]
            x_pad[i, :L] = torch.from_numpy(s)
            mask[i, :L] = True
        return x_pad, mask

    x_e, m_e = pad_mod(eye_seqs)
    x_p, m_p = pad_mod(phy_seqs)
    y = torch.tensor(labels, dtype=torch.long)
    return x_e, m_e, x_p, m_p, y


class MultiModalAttentionModel(nn.Module):
    """
    Encoders separados + atención cruzada simple.
    """
    def __init__(self, eye_dim, phy_dim, num_classes,
                 d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.eye_proj = nn.Linear(eye_dim, d_model)
        self.phy_proj = nn.Linear(phy_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.eye_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.phy_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.cls_head = nn.Linear(2*d_model, num_classes)

    def pooled(self, h, mask):
        mask_f = mask.unsqueeze(-1).float()
        h_sum = (h * mask_f).sum(dim=1)
        lengths = mask_f.sum(dim=1).clamp(min=1.0)
        return h_sum / lengths

    def forward(self, x_e, m_e, x_p, m_p):
        # encode cada modalidad
        key_pad_eye = ~m_e
        key_pad_phy = ~m_p

        h_e = self.eye_enc(self.eye_proj(x_e),
                           src_key_padding_mask=key_pad_eye)
        h_p = self.phy_enc(self.phy_proj(x_p),
                           src_key_padding_mask=key_pad_phy)

        # atención cruzada: eye (query) sobre physio (key/value)
        # usamos un "token global" promedio como query
        eye_global = self.pooled(h_e, m_e).unsqueeze(1)  # [B,1,D]
        attn_out, _ = self.cross_attn(
            query=eye_global,
            key=h_p,
            value=h_p,
            key_padding_mask=key_pad_phy
        )
        attn_out = attn_out.squeeze(1)  # [B,D]

        # concatenar representaciones globales
        h_eye = self.pooled(h_e, m_e)
        fused = torch.cat([h_eye, attn_out], dim=-1)
        logits = self.cls_head(fused)
        return logits


# -----------------------------
# 6) EXTRACCIÓN DE FRAMES FACIALES Y ViT
# -----------------------------

def extract_frames(video_path, out_dir, fps=1):
    """
    Extrae frames cada (1/fps) segundos.
    Asume que luego tú organizas carpetas por clase/etiqueta.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 0:
        vid_fps = 30.0
    step = max(int(round(vid_fps / fps)), 1)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # TODO: opcionalmente recortar cara aquí
            out_path = out_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Guardados {saved} frames en {out_dir}")


class FaceFramesViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = (
            models.ViT_B_16_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        self.vit = models.vit_b_16(weights=weights)
        in_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.vit(x)


def train_vit_faceframes(data_root, num_classes, batch_size=32,
                         epochs=10, lr=3e-4, device="cuda"):
    """
    data_root:
      ├── class_0
      │    ├── img_0.jpg
      ├── class_1
      ...
    """
    data_root = Path(data_root)

    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    tfm = weights.transforms()

    train_ds = datasets.ImageFolder(root=str(data_root), transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FaceFramesViT(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_y, all_pred = [], []

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=-1)
            all_y.extend(yb.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

        f1 = f1_score(all_y, all_pred, average="macro")
        print(f"[ViT] Epoch {epoch} loss={running_loss/len(train_ds):.4f} "
              f"F1-macro={f1:.3f}")

    print("Reporte final ViT:")
    print(classification_report(all_y, all_pred))


# -----------------------------
# 7) TRANSFERENCIA A DATOS MINEROS
# -----------------------------

def build_transfer_model(emotion_model: nn.Module, new_num_classes: int):
    """
    Congela el encoder y cambia solo la cabeza de clasificación
    para datos mineros (p.ej. estrés, fatiga, etc.).
    """
    # ejemplo para TemporalTransformer; ajusta según tu clase
    for param in emotion_model.parameters():
        param.requires_grad = False

    in_dim = emotion_model.cls_head.in_features
    emotion_model.cls_head = nn.Linear(in_dim, new_num_classes)
    # solo entrena nueva cabeza
    for p in emotion_model.cls_head.parameters():
        p.requires_grad = True
    return emotion_model


# -----------------------------
# EJEMPLO DE USO INTEGRADO
# -----------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) cargar eye-tracking
    df_eye = load_eyetracking_csv("data/vreed/eye_tracking")
    feat_cols_eye = get_feature_cols(df_eye)

    # -------- TABULAR UNIMODAL 4-FACTOR --------
    X = df_eye[feat_cols_eye].values.astype(np.float32)
    y4 = df_eye[LABEL_4F].values.astype(int)
    run_tabular_kfold(X, y4, task_name="4F_tabular")

    # -------- SECUENCIAL TRANSFORMER UNIMODAL --------
    eye_seqs, labels_seq = build_sequences(df_eye, feat_cols_eye, LABEL_4F)
    num_classes_4f = len(np.unique(labels_seq))
    train_transformer_5fold(
        eye_seqs, labels_seq, num_classes=num_classes_4f,
        device=device
    )

    # A partir de aquí puedes:
    # - Repetir para arousal/valence (cambiando label_col)
    # - Construir df_phys, seqs multimodales y entrenar MultiModalAttentionModel
    # - Extraer frames por video y entrenar train_vit_faceframes(...)
    # - Usar build_transfer_model(...) con datos mineros

if __name__ == "__main__":
    main()
