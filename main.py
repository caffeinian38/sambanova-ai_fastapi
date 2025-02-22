from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import pytesseract

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define Upload Directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pytesseract
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Set path if needed

def preprocess_image(image_path):
    img = cv2.imread(str(image_path))  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  
    return Image.fromarray(thresh)  

def extract_text_from_image(image_path):
    image = preprocess_image(image_path)  
    text = pytesseract.image_to_string(image, config='--psm 6')  
    return text.strip()

# Image Upload & Text Extraction Endpoint
@app.post("/upload")
async def upload_and_extract(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from image
        extracted_text = extract_text_from_image(file_path)
        print(extracted_text)

        return {"filename": file.filename, "extracted_text": extracted_text}

    except Exception as e:
        return {"error": str(e)}

# Chatbot Endpoint
class ChatRequest(BaseModel):
    userInput: str

@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    userInput = request.userInput
    apiKey = os.getenv('AI_API_KEY')
    baseUrl = os.getenv('AI_BASE_URL')

    try:
        client = openai.OpenAI(api_key=apiKey, base_url=baseUrl)

        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=[{"role": "system", "content": "You are a helpful Health Assistant"},
                      {"role": "user", "content": userInput}],
            temperature=0.1,
            top_p=0.1
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}



@app.get('/')
def start():
    return {'success': True, 'message':'Hello'}
