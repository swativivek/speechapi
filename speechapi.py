from flask import Flask, request, jsonify
from flask_cors import CORS
import torchaudio
import librosa
import numpy as np
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load Hugging Face ASR pipeline for speech-to-text
logger.info("Loading ASR model...")
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")

# Load T5 model for grammar correction
logger.info("Loading T5 model...")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def transcribe_audio(file_path):
    try:
        logger.info("Transcribing audio...")
        start_time = time.time()
        
        # Load and preprocess audio
        audio, sampling_rate = torchaudio.load(file_path, format="wav")
        audio = audio.numpy()[0]
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        
        # Transcribe audio
        transcript = asr_pipeline(audio)["text"]
        
        logger.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return f"Error transcribing audio: {str(e)}"

def correct_grammar(text):
    try:
        logger.info("Correcting grammar...")
        start_time = time.time()
        
        # Correct grammar
        input_text = f"grammar: {text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Grammar correction completed in {time.time() - start_time:.2f} seconds")
        return corrected_text
    except Exception as e:
        logger.error(f"Error correcting grammar: {str(e)}")
        return f"Error correcting grammar: {str(e)}"

@app.route("/correct-audio", methods=["POST"])
def correct_audio():
    if "file" not in request.files:
        logger.error("No file provided")
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        logger.error("Empty file provided")
        return jsonify({"error": "Empty file provided"}), 400

    # Save the uploaded file temporarily
    file_path = "temp_audio.wav"
    audio_file.save(file_path)
    logger.info(f"Audio file saved to {file_path}")

    # Transcribe audio
    transcript = transcribe_audio(file_path)
    if "Error" in transcript:
        os.remove(file_path)
        logger.error(f"Transcription error: {transcript}")
        return jsonify({"error": transcript}), 500

    # Correct grammar
    corrected_text = correct_grammar(transcript)
    os.remove(file_path)
    logger.info("Temporary audio file removed")

    return jsonify({"corrected_text": corrected_text})

if __name__ == "__main__":
    app.run(debug=True)