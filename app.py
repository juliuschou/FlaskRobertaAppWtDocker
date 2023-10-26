from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = Flask(__name__)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

@app.route("/predict", methods=["POST"])
def predict():
    input_ids = torch.tensor(
        tokenizer.encode(request.json[0], add_special_tokens=True)
    ).unsqueeze(0)

    # Corrected the condition to convert tensor to numpy
    if input_ids.requires_grad:
        numpy_func = input_ids.detach().cpu().numpy()
    else:
        numpy_func = input_ids.cpu().numpy()

    # Corrected the variable name and removed unnecessary space
    inputs = {session.get_inputs()[0].name: numpy_func}

    out = session.run(None, inputs)

    # Corrected the space in np.argmax
    result = np.argmax(out)

    # Corrected the return statement
    return jsonify({"positive": bool(result)})

# Corrected the condition for running the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

