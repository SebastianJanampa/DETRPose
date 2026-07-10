import os
import glob
import cv2
import numpy as np
from PIL import Image
from typing import Generator, Any


class InputHandler:
    @staticmethod
    def to_pil_image(inputs: Any) -> Image.Image:
        """Converts PIL or OpenCV image inputs to an RGB PIL Image."""
        if isinstance(inputs, Image.Image):
            return inputs.convert("RGB")
        elif isinstance(inputs, np.ndarray):
            # Convert OpenCV BGR to PIL RGB
            return Image.fromarray(cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB))
        raise TypeError("Expected a PIL Image or numpy ndarray.")

    @staticmethod
    def process_video(video_path: str) -> Generator[dict, None, None]:
        """Yields raw frames and indices from a video file."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        # Get video properties for saving later if needed
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield {"frame_idx": frame_idx, "frame_data": frame, "fps": fps, "orig_size": (w, h)}
            frame_idx += 1
        cap.release()

    @staticmethod
    def process_folder(folder_path: str) -> Generator[dict, None, None]:
        """Yields file paths and detected types within a directory."""
        paths = list(glob.iglob(f"{folder_path}/*.*"))
        for p in paths:
            ext = os.path.splitext(p)[-1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                yield {"path": p, "type": "image"}
            elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
                yield {"path": p, "type": "video"}