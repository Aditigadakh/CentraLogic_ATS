import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import torch
from pydantic import BaseModel
import json

# Initialize FastAPI app
app = FastAPI()

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model initialization
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Processor initialization
processor = AutoProcessor.from_pretrained(model_id)

# Pipeline initialization for ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=10,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Pipeline initialization for summarization
summarizer = pipeline("summarization")

# Pydantic model for request body validation
class AudioFile(BaseModel):
    file: UploadFile

# Endpoint to transcribe audio file
@app.post("/transcribe_audio/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded audio file locally
        audio_path = save_uploaded_file(audio_file)

        # Perform speech recognition on the audio file
        result = pipe(audio_path)

        # Extract the transcribed text from the result
        transcription = result[0]['text']

        # Summarize the transcription
        summary = summarize_text(transcription)

        # Extract timestamps
        timestamps_result = pipe(audio_path, return_timestamps="word")
        timestamps = timestamps_result["chunks"]

        # Save transcription, summary, and timestamps to text files
        save_transcription(transcription)
        save_summary(summary)
        save_timestamps(timestamps)

        return {"transcription": transcription, "summary": summary, "timestamps": timestamps}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to save uploaded audio file locally
def save_uploaded_file(audio_file: UploadFile) -> str:
    # Create a directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", audio_file.filename)
    with open(file_path, "wb") as f:
        f.write(audio_file.file.read())
    return file_path

# Function to save transcription to a text file
def save_transcription(transcription: str):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "transcription.txt"), "w") as f:
        f.write(transcription)

# Function to summarize text and save to a text file
def save_summary(summary: str):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.txt"), "w") as f:
        f.write(summary)

# Function to save timestamps to a JSON file
def save_timestamps(timestamps: List[dict]):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "timestamps.txt"), "w") as f:
        json.dump(timestamps, f)

# Function to summarize text
def summarize_text(text: str) -> str:
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Run the FastAPI server with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
