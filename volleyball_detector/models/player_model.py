import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

class Player:
    def __init__(self, frame_num, team, confidence, coordinates, image):
        self.frame_num = frame_num
        self.team = team
        self.confidence = confidence
        self.x1, self.y1, self.x2, self.y2 = coordinates
        self.image = image  # NumPy image (OpenCV format)
        self.jersey_number = None
        self.ocr_confidence = 0.0
        self.needs_review = True

    def image_to_base64(self):
        """Convert image to base64 for HTML display"""
        # Convert BGR (OpenCV) to RGB
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
