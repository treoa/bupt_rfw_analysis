import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

from rich import print


# Load the model and processor from Hugging Face Hub
model_name = "prithivMLmods/Gender-Classifier-Mini"
processor = AutoImageProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name)

def classify_gender(image_path):
    """
    Classifies the gender from an image file.
    Args:
        image_path (str): The path to the input image.
    Returns:
        dict: A dictionary with predicted labels and their probabilities.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": str(e)}

    # Process the image and prepare it for the model
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    labels = model.config.id2label
    predictions = {labels[i]: round(probs[i], 4) for i in range(len(probs))}

    return predictions

# Example usage:
result = classify_gender("./test.jpg")
print(result)
# Expected output: {'Female ♀': 0.985, 'Male ♂': 0.015} (example values)