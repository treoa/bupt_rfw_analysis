import os
import sys
import cv2
import torch
import warnings

warnings.filterwarnings('ignore')

import torch.nn as nn
import torch.nn.functional as F 

from glob import glob
from PIL import Image
from torchvision import transforms

from mivolo.predictor import Predictor

def predict_gender(img_path: str, detector_weights="weights/yolov8x_person_face.pt", checkpoint="weights/mivolo_imdb.pth.tar"):
    """
    Predict the gender of a person in an image using the MiVOLO model.

    Args:
        image_path (str): Path to the input image file.
        detector_weights (str, optional): Path to the YOLOv8 detector weights file.
            Defaults to "models/yolov8x_person_face.pt".
        checkpoint (str, optional): Path to the MiVOLO model checkpoint file.
            Defaults to "models/mivolo_imbd.pth.tar".

    Returns:
        str: Predicted gender ("male" or "female") of the first detected person,
             or "No person detected" if no person is found.

    Raises:
        ValueError: If the image cannot be loaded from the provided path.
    """
    class Args:
        def __init__(self):
            self.detector_weights = detector_weights
            self.checkpoint = checkpoint
            # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.with_persons = True,
            self.disable_faces = False,
            self.draw = True,
    os.environ["YOLO_VERBOSE"] = "true"
    # Instantiate the Predictor with the arguments
    args = Args()
    predictor = Predictor(args)

    # Load the image using OpenCV
    image = cv2.imread(img_path)
    if not image.any():
        raise ValueError("Could not load image")

    # Perform prediction
    detected_obj, out_image = predictor.recognize(image)

    # Check if any persons were detected
    if not detected_obj:
        return "No person detected"
    
    # save the out image as tst_drawn.jpg
    cv2.imwrite("tst_drawn.jpg", out_image)
    
    # traverse and get the gender and skip None values
    for obj in detected_obj.genders:
        if obj is not None:
            return obj.capitalize()
    
    # return detected_obj

    # Get the gender from the first detection
    # detection = results[0]
    # return detection.genders

gender = predict_gender("test.jpg")
print(gender)

