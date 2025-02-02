from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import soundfile as sf
import numpy as np
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os

app = Flask(__name__)
CORS(app)

# Load models lazily (only when needed)
def load_asr_pipeline():
    return pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")  # Use a smaller model

def load_grammar_correction_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def transcribe_audio(file_path):
    try:
        # Load audio using librosa (more memory-efficient)
        audio, sampling_rate = librosa.load(file_path, sr=16000)  # Resample to 16kHz
        transcript = load_asr_pipeline()(audio)["text"]
        return transcript
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def correct_grammar(text):
    tokenizer, model = load_grammar_correction_model()
    input_text = f"grammar: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

@app.route("/correct-audio", methods=["POST"])
def correct_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty file provided"}), 400

    file_path = "temp_audio.wav"
    audio_file.save(file_path)

    transcript = transcribe_audio(file_path)
    if "Error" in transcript:
        os.remove(file_path)
        return jsonify({"error": transcript}), 500

    corrected_text = correct_grammar(transcript)
    os.remove(file_path)

    return jsonify({"corrected_text": corrected_text})

if __name__ == "__main__":
    app.run(debug=True)