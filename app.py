import os
import tempfile
from flask import Flask, request, jsonify
from Predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Medicine API is running"})

@app.route("/predict", methods=["POST"])
def predict_medicine():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send as multipart/form-data with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        raw, snapped = predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    return jsonify({
        "raw_prediction": raw,
        "medicine": snapped
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
