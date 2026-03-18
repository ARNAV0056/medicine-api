import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import process as fuzz_process
from huggingface_hub import snapshot_download
import os

# Download model from Hugging Face on first run (cached after that)
print("Downloading / loading model from Hugging Face...")

HF_TOKEN = os.environ.get("HF_TOKEN", None)  # only needed if your HF repo is Private

MODEL_PATH = snapshot_download(
    repo_id="Arnav-Vashist/medicine-ocr",
    force_download=True
)

print(f"Model path: {MODEL_PATH}")

# Load processor from original base model (more reliable)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# Load your fine-tuned model weights
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded on {device}")


MEDICINE_VOCAB = [
    "Ace", "Aceta", "Alatrol", "Amodis", "Atrizin",
    "Axodin", "Az", "Azithrocin", "Azyth", "Bacaid",
    "Backtone", "Baclofen", "Baclon", "Bacmax", "Beklo",
    "Bicozin", "Canazole", "Candinil", "Cetisoft", "Conaz",
    "Dancel", "Denixil", "Diflu", "Dinafex", "Disopan",
    "Esonix", "Esoral", "Etizin", "Exium", "Fenadin",
    "Fexo", "Fexofast", "Filmet", "Fixal", "Flamyd",
    "Flexibac", "Flexilax", "Flugal", "Ketocon", "Ketoral",
    "Ketotab", "Ketozol", "Leptic", "Lucan-R", "Lumona",
    "M-Kast", "Maxima", "Maxpro", "Metro", "Metsina",
    "Monas", "Montair", "Montene", "Montex", "Napa",
    "Napa Extend", "Nexcap", "Nexum", "Nidazyl", "Nizoder",
    "Odmon", "Omastin", "Opton", "Progut", "Provair",
    "Renova", "Rhinil", "Ritch", "Rivotril", "Romycin",
    "Rozith", "Sergel", "Tamen", "Telfast", "Tridosil",
    "Trilock", "Vifas", "Zithrin",
]


def predict(image_path, use_vocab_snap=True):
    image        = Image.open(image_path).convert("RGB")
    pixel_values = processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            num_beams      = 4,
            max_new_tokens = 64,
            early_stopping = True,
        )
    raw_pred = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    if use_vocab_snap:
        match   = fuzz_process.extractOne(
            raw_pred, MEDICINE_VOCAB, score_cutoff=60
        )
        snapped = match[0] if match else raw_pred
        return raw_pred, snapped

    return raw_pred, raw_pred
