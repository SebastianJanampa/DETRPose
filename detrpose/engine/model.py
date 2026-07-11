import os
import re
import cv2
import math
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf

from detrpose.core import LazyConfig
from detrpose.core.lazy import _visit_dict_config
from detrpose.core.utils import _convert_target_to_string

from .hf_model import HFModel
from .io import InputHandler
from .annotators import COCOVisualizer, CrowdPoseVisualizer


def _resolve_config_path(model_name_or_path: str) -> str:
    """
    Resolves the path to the configuration file.
    Checks the current working directory first, then searches in detrpose/configs.
    """
    if os.path.isfile(model_name_or_path):
        return model_name_or_path

    if not model_name_or_path.endswith(('.py')):
        model_name = f"{model_name_or_path}.py"
    else:
        model_name = model_name_or_path

    if os.path.isfile(model_name):
        return model_name

    current_dir = os.path.dirname(os.path.abspath(__file__))
    detrpose_dir = os.path.dirname(current_dir)
    configs_dir = os.path.join(detrpose_dir, "configs")

    if os.path.isdir(configs_dir):
        for root, dirs, files in os.walk(configs_dir):
            if model_name in files:
                return os.path.join(root, model_name)

    raise FileNotFoundError(
        f"Configuration for '{model_name_or_path}' could not be found locally "
        f"or in the detrpose/configs directory."
    )


class DETR(object):
    def __init__(self, model:str, device=None, resume:str|None=None, deploy=True):
        super().__init__()
        config_path = _resolve_config_path(model)
        self.cfg = LazyConfig.load(config_path)
        self._array_counter = 1 # Keep track of sequential image numbers for numpy arrays / PIL inputs

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.create_model(resume, deploy=deploy)

        num_keypoints = self.model.model.transformer.num_keypoints

        if num_keypoints == 17:
            self.annotator = COCOVisualizer()
            self.annotator_type = "COCO"
        elif num_keypoints == 14:
            self.annotator = CrowdPoseVisualizer()
            self.annotator_type = "CrowdPose"

    def create_model(self, resume:str|None=None, deploy:bool=True):
        """
        :param resume: path for pretrained model
        :param deploy: set to False only to export HF model card
        """
        def _replace_type_by_name(x):
            if "_target_" in x and callable(x._target_):
                try:
                    x._target_ = _convert_target_to_string(x._target_)
                except AttributeError:
                    pass

        cfg_copy = deepcopy(self.cfg)
        _visit_dict_config(cfg_copy, _replace_type_by_name)
        hf_config = {
            'model': OmegaConf.to_container(cfg_copy.model, resolve=True),
            'postprocessor': OmegaConf.to_container(cfg_copy.postprocessor, resolve=True)
        }

        # 2. Instantiate and load weights
        if resume is None:
            assert hasattr(self.cfg.public, 'hf_model')
            # Explicitly pass your safe, local config to Hugging Face!
            # HF will use this config to build the model, and then safely download/apply the weights.
            model = HFModel.from_pretrained(self.cfg.public.hf_model, config=hf_config)
        else:
            model = HFModel(hf_config)
            if 'pt' in resume or 'pth' in resume:
                ckpt = torch.load(resume, map_location='cpu')
                if 'ema' in ckpt:
                    model.model.load_state_dict(ckpt['ema'])
                else:
                    model.model.load_state_dict(ckpt['model'])
        if deploy:
            model = model.deploy()
        model.eval()
        self.model = model.to(self.device)

    def _format_img_size(self, img_size):
        """Converts integer to tuple and pads dimensions to be divisible by 32."""
        if isinstance(img_size, int):
            size_tuple = (img_size, img_size)
        else:
            size_tuple = (img_size[0], img_size[1])

        # Formula to round up to the nearest multiple of 32
        w_new = int(math.ceil(size_tuple[0] / 32.0)) * 32
        h_new = int(math.ceil(size_tuple[1] / 32.0)) * 32
        return (h_new, w_new)

    def _run_inference(self, im_pil: Image.Image, thrh: float, img_size: tuple[int], **kwargs) -> dict:
        """Executes the forward pass, filters thresholds, plots, and saves."""
        # Extract kwargs
        save = kwargs.get("save", False)
        save_dir = kwargs.get("save_dir", "output")
        show = kwargs.get("show", False)
        exist_ok = kwargs.get("exist_ok", False)
        filename = kwargs.get("filename", '')

        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        target_size = self._format_img_size(img_size)

        transforms = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.model(im_data)
            results = self.model.postprocessor(outputs, orig_size)

        scores, labels, keypoints = results
        scores = scores[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()
        keypoints = keypoints[0].detach().cpu().numpy()

        idx = scores > thrh
        valid_scores, valid_labels, valid_keypoints = scores[idx], labels[idx], keypoints[idx]

        # Visualizer Logic
        im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        self.annotator.draw_on(im_cv2, valid_keypoints)

        # Show Logic
        if show:
            cv2.imshow(f"DETRPose Inference - {filename}", im_cv2)
            cv2.waitKey(0) # Pauses until you press a key

        # Save Logic
        if save:
            save_path = os.path.join(save_dir, filename) + '.png'
            cv2.imwrite(save_path, im_cv2)
            print(f"Annotated image saved to: {save_path}")

        return {
            "scores": valid_scores, "labels": valid_labels, "keypoints": valid_keypoints,
            "original_size": (w, h), "annotated_image": im_cv2
        }

    def _process_video(self, video_path: str, thrh: float, img_size: tuple, **kwargs):
        print(f"Processing video: {video_path}")

        save = kwargs.get("save", False)
        save_dir = kwargs.get("save_dir", "output")
        show = kwargs.get("show", False)
        exist_ok = kwargs.get("exist_ok", True)
        filename = kwargs.get("filename", '')

        if save:
            os.makedirs(save_dir, exist_ok=exist_ok)
            save_path = os.path.join(save_dir, filename)

        out = None
        for frame_entry in InputHandler.process_video(video_path):
            if out is None and save:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = frame_entry["fps"]
                w, h = frame_entry["orig_size"]
                out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

            pil_img = InputHandler.to_pil_image(frame_entry["frame_data"])

            # Override inner kwargs so inference doesn't block (show) or spam images (save)
            inner_kwargs = kwargs.copy()
            inner_kwargs["save"] = False
            inner_kwargs["show"] = False

            res = self._run_inference(pil_img, thrh, img_size, **inner_kwargs)
            annotated_frame = res["annotated_image"]

            if show:
                cv2.imshow(f"Video: {filename}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save and out is not None:
                out.write(annotated_frame)

        if out is not None:
            out.release()
            print(f"Annotated video saved to: {save_path}")

        if show:
            cv2.destroyAllWindows()

        return {"file": video_path}

    def _process_folder(self, folder_path: str, thrh: float, img_size: tuple, **kwargs):
        print(f"Processing folder: {folder_path}")

        all_results = []
        for item in InputHandler.process_folder(folder_path):
            ref = self(item["path"], thrh, img_size, **kwargs)
            all_results.append(ref)

        return all_results

    def _run_live_webcam(self, camera_id: int, thrh: float, img_size: tuple, **kwargs):
        """Opens a live webcam feed, runs inference, and displays the results in real-time."""
        print(f"Starting live webcam feed (Camera ID: {camera_id}). Press 'q' in the window to quit.")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam with ID {camera_id}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting...")
                break
            frame = cv2.flip(frame, 1)

            # Pass save=False so we don't save thousands of JPEGs
            inner_kwargs = kwargs.copy()
            inner_kwargs["save"] = False
            inner_kwargs["show"] = False

            # Convert the raw OpenCV BGR frame to a PIL RGB Image for inference
            pil_img = InputHandler.to_pil_image(frame)

            # Run inference (pass save_path=None so we don't save thousands of JPEGs)
            res = self._run_inference(pil_img, thrh, img_size, **inner_kwargs)

            # Display the annotated image in a live pop-up window
            cv2.imshow("DETRPose Live Inference", res["annotated_image"])

            # Listen for the 'q' key to stop the webcam feed (1ms delay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up when done
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam feed closed.")

    @staticmethod
    def check_folder(output_dir, exist_ok=False):
        """
        Creates an output directory, incrementing a numeric suffix if the path already exists.

        For example, if 'outputs/train/exp1' exists, it will create 'outputs/train/exp2'.
        If 'outputs/train/exp' is given, it will start by checking for 'outputs/train/exp1'.

        Args:
            output_dir (str): The desired directory path, e.g., 'outputs/train/exp1'.
            exist_ok (bool): If True, it will use the provided `output_dir` as is,
                             creating it if it doesn't exist, without searching for a new path.

        Returns:
            str: The path to the created directory.
        """
        if exist_ok:
            os.makedirs(output_dir, exist_ok=True)
            return output_dir

        # Separate the base path from any existing numeric suffix
        # e.g., 'predictions/run1' -> base='predictions/run', num=1
        match = re.match(r'(.*?)(\d*)$', output_dir)
        base, num_str = match.groups()

        # If the original path had no number, start counting from 1.
        # Otherwise, start from the number that was found.
        counter = int(num_str) if num_str else 1

        current_dir = f"{base}{counter}"

        # Increment the counter until a non-existent path is found
        while os.path.exists(current_dir):
            counter += 1
            current_dir = f"{base}{counter}"

        os.makedirs(current_dir, exist_ok=True)
        return current_dir

    def __call__(
            self,
            source, thrh: float = 0.5,
            img_size: int | tuple = 640,
            **kwargs,
    ):
        kwargs.setdefault("show", False)
        kwargs.setdefault("save", False)
        kwargs.setdefault("exist_ok", False)
        kwargs.setdefault("save_dir", Path(self.check_folder(
            kwargs['save_dir'] if hasattr(kwargs, 'save_dir') else 'predictions/runs1', kwargs['exist_ok'])
        ) if kwargs['save'] else None)

        # 1. Handle Webcams (If input is an integer 0, or string "0")
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cam_id = int(source)
            return self._run_live_webcam(cam_id, thrh, img_size, **kwargs)

        # 2. Handle PIL and OpenCV Images
        elif isinstance(source, (Image.Image, np.ndarray)):
            pil_img = InputHandler.to_pil_image(source)
            filename = f"image{self._array_counter}.jpg"
            self._array_counter += 1
            kwargs['filename'] = [filename]
            return self._run_inference(pil_img, thrh, img_size, **kwargs)

        # 3. Handle File and Folder Paths
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Path '{source}' not found.")

            if os.path.isdir(source):
                return self._process_folder(source, thrh, img_size, **kwargs)

            ext = os.path.splitext(source)[-1].lower()
            filename = os.path.splitext(os.path.basename(source))[0]
            kwargs['filename'] = filename

            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                return self._run_inference(Image.open(source).convert("RGB"), thrh, img_size, **kwargs)
            elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
                return self._process_video(source, thrh, img_size, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {source}")

        raise TypeError("Invalid input type. Must be PIL, OpenCV array, valid path, or camera ID (0).")