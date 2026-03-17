Medicine OCR API
A REST API that identifies medicine names from images using a fine-tuned TrOCR (Vision Encoder-Decoder) model.
Model

Architecture: TrOCR — ViT image encoder + GPT-2 text decoder
Task: Reading medicine names from images (OCR)
Accuracy: 97.51%
Vocab: 78 medicine names with fuzzy matching to correct minor OCR errors
Model hosted on: Hugging Face
Stack

Python, Flask, PyTorch, Hugging Face Transformers
Deployed on Render
Model weights hosted on Hugging Face Hub
