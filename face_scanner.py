# face_scanner.py
import cv2
import numpy as np
import base64
from deepface import DeepFace

def decode_base64_image(base64_string):
    """Converts a base64 string from the frontend into an OpenCV image array."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def match_face_in_db(live_frame_b64, db_visitors):
    """
    Compares the live frame face against all visitor photos in the database
    using DeepFace (Facenet model).
    """
    live_img = decode_base64_image(live_frame_b64)
    if live_img is None:
        return None

    for visitor in db_visitors:
        db_photo_b64 = visitor.get("visitorPhoto")
        if not db_photo_b64:
            continue
            
        db_img = decode_base64_image(db_photo_b64)
        if db_img is None:
            continue
            
        try:
            # DeepFace compares the two images. 
            # enforce_detection=False prevents crashes if a face isn't perfectly visible in one frame
            result = DeepFace.verify(
                img1_path=live_img, 
                img2_path=db_img, 
                model_name="Facenet", 
                enforce_detection=False
            )
            
            if result["verified"]:
                print(f"Face matched with visitor: {visitor.get('visitorName')}")
                return visitor # Match found!
                
        except Exception as e:
            print(f"DeepFace verification error for a visitor: {e}")
            continue
            
    return None