import os
import uuid
import time
import torch
from PIL import Image
from torchvision import transforms

from model import model, device
from preprocess import generate_mel_spectrogram

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["Generated", "Human"]


def predict_audio(file):

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("spectrograms", exist_ok=True)

    audio_path = f"uploads/{uuid.uuid4()}.wav"
    image_path = f"spectrograms/{uuid.uuid4()}.png"

    start = time.time()

    with open(audio_path, "wb") as buffer:
        buffer.write(file.file.read())

    print(f"Saving audio: {time.time() - start:.2f} sec")

    start = time.time()

    generate_mel_spectrogram(audio_path, image_path)

    print(f"Generating spectrogram: {time.time() - start:.2f} sec")

    start = time.time()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    print(f"Loading image: {time.time() - start:.2f} sec")

    start = time.time()

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    print(f"Model prediction: {time.time() - start:.2f} sec")

    prediction = classes[predicted.item()]
    confidence = float(confidence.item() * 100)

    os.remove(audio_path)
    os.remove(image_path)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
