"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import sys
import glob

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image, ImageDraw
from copy import deepcopy
from annotators import COCOVisualizer, CrowdPoseVisualizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import LazyConfig, instantiate

annotators = {'COCO': COCOVisualizer, 'CrowdPose': CrowdPoseVisualizer}

def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    annotator = annotators[annotator_type]()

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)


    scores, labels, keypoints = output
    scores = scores[0].detach().cpu().numpy()
    keypoints = keypoints[0].detach().cpu().numpy()
    
    # Filter by score
    idx = scores > thrh
    valid_keypoints = keypoints[idx] # Shape: (N, K, 2)
    
    im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    annotator.draw_on(im_cv2, valid_keypoints)
    cv2.imwrite(f"{OUTPUT_NAME}.jpg", im_cv2)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{OUTPUT_NAME}.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    annotator = annotators[annotator_type]()
    
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)


        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)

        scores, labels, keypoints = output
        scores = scores[0].detach().cpu().numpy()
        keypoints = keypoints[0].detach().cpu().numpy()
        
        # Filter by score
        idx = scores > thrh
        valid_keypoints = keypoints[idx] # Shape: (N, K, 2)
        
        im_cv2 = frame
        annotator.draw_on(im_cv2, valid_keypoints)
        cv2.imwrite(f"{OUTPUT_NAME}.jpg", im_cv2)

        im_cv2 = np.ascontiguousarray(im_cv2)  # Ensure contiguous memory layout for VideoWriter
        out.write(im_cv2)  # Write annotated frame to output video

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

def process_file(model, device, file_path):
    # Check if the input file is an image or a vide
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)
        print("Video processing complete.")

def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)

def main(args):
    # Global variable
    global OUTPUT_NAME, thrh, annotator_type

    """Main function"""
    cfg = LazyConfig.load(args.config)

    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)

    num_body_points = model.transformer.num_body_points 
    if  num_body_points == 17:
        annotator_type = 'COCO'
    elif num_body_points == 14:
        annotator_type = 'CrowdPose'
    else:
        raise Exception(f'Not implemented annotator for model with {num_body_points} keypoints')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    thrh = 0.5 if args.thrh is None else args.thrh

    # Check if the input argumnet is a file or a folder
    file_path = args.input
    if os.path.isdir(file_path):
        # Process a folder
        folder_dir = args.input
        output_dir = f"{folder_dir}/output"
        os.makedirs(output_dir, exist_ok=True)
        paths = list(glob.iglob(f"{folder_dir}/*.*"))
        for file_path in paths:
            OUTPUT_NAME = file_path.replace(f'{folder_dir}/', f'{output_dir}/').split('.')[0]
            OUTPUT_NAME = f"{OUTPUT_NAME}_{annotator_type}"
            process_file(model, device, file_path)
    else:
        # Process a file
        OUTPUT_NAME = f'torch_results_{annotator_type}'
        process_file(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-t", "--thrh", type=float, required=False, default=None)
    args = parser.parse_args()
    main(args)
