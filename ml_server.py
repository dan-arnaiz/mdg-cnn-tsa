from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np

app = Flask(__name__)

# Load artifacts (trained under sklearn 1.7)
scaler = joblib.load("preprocessor.joblib")
selector = joblib.load("selector.joblib")

class CNN_TSA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(45, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = CNN_TSA()
model.load_state_dict(torch.load("cnn_tsa_model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    X = np.array(data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    with torch.no_grad():
        tensor_input = torch.tensor(X_selected, dtype=torch.float32)
        output = model(tensor_input).item()

    return jsonify({"prediction": float(output)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)