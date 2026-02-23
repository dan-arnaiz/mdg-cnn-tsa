from flask import Flask, request, jsonify
import joblib
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH      = os.path.join(BASE_DIR, "models/cnn_tsa/baseline_model/main/standard_k45/best_weights.pt")
CONFIG_PATH     = os.path.join(BASE_DIR, "models/cnn_tsa/baseline_model/main/standard_k45/config.json")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessing_output/v1_std_corr90_k45_w48s24/preprocessor.joblib")
SELECTOR_PATH   = os.path.join(BASE_DIR, "preprocessing_output/v1_std_corr90_k45_w48s24/selector.joblib")
METADATA_PATH   = os.path.join(BASE_DIR, "preprocessing_output/v1_std_corr90_k45_w48s24/preprocess_metadata.json")


# ── Model architecture (must match training exactly) ──────────────────────────
class CNNTSA(nn.Module):
    def __init__(self, num_features=39, hidden_dim=64, num_heads=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.mhsa  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1   = nn.Linear(hidden_dim, 128)
        self.fc2   = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        attn_out, _ = self.mhsa(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))


# ── Load artifacts ─────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
    
def sanitize_array(x):
    # Placeholder implementation.
    # This must match the original training logic if possible.
    return x

preprocessor = joblib.load(PREPROCESSOR_PATH)
selector     = joblib.load(SELECTOR_PATH)

with open(METADATA_PATH) as f:
    metadata = json.load(f)
selected_features = metadata["selected_features"]

model = CNNTSA(
    num_features=cfg["num_features"],
    hidden_dim=cfg["hidden_dim"],
    num_heads=cfg["num_heads"]
)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

print(f"[ml_server] Model loaded - features={cfg['num_features']}, "
      f"hidden={cfg['hidden_dim']}, heads={cfg['num_heads']}")


# ── Inference endpoint ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body: { "features": { "Flow Duration": 1234, "Protocol": 6, ... } }
    Keys must match the column names in preprocess_metadata.json (without 'num__' prefix).
    Returns: { "prediction": 0.83, "label": 1 }
    """
    try:
        raw = request.json.get("features")
        if raw is None:
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        # Build DataFrame with exact training column names
        df = pd.DataFrame([raw])[selected_features]

        # Apply preprocessor (StandardScaler / ColumnTransformer)
        Xt = preprocessor.transform(df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        # Apply SelectKBest
        Xt_sel = selector.transform(Xt)   # shape: (1, 39)

        # Reshape for CNN-TSA: (1, 39) → (1, 39, 1)
        Xt_tensor = torch.tensor(Xt_sel, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            prob = model(Xt_tensor).item()   # sigmoid already in forward()

        label = 1 if prob >= 0.50 else 0
        return jsonify({"prediction": round(prob, 6), "label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)