
<h2 align="center">
   DETRPose: Real-Time End-to-End Multi-Person Pose Estimation via Modified Transformer Decoder and Novel Denoising Keypoints
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/DETRPose/blob/inference_only/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="https://www.arxiv.org/abs/2506.13027">
        <img alt="arxiv" src="https://img.shields.io/badge/-paper-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>

  <a href="https://huggingface.co/collections/SebasJanampa/detrpose">
      <img src='https://img.shields.io/badge/-COLLECTIONS-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF5500&color=orange'>
  </a>
   
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://www.arxiv.org/abs/2506.13027">    DETRPose: Real-Time End-to-End Multi-Person Pose Estimation via Modified Transformer Decoder and Novel Denoising Keypoints
</a>
</p>

<p align="center">
Sebastian Janampa and Marios Pattichis
</p>

<p align="center">
The University of New Mexico
  <br>
Department of Electrical and Computer Engineering
</p>

---


DETRPose is the first real-time end-to-end transformer model for multi-person pose estimation, 
achieving outstanding results on the COCO and CrowdPose datasets. In this work, we propose a 
new denoising technique suitable for pose estimation that uses the Object Keypoint Similarity (OKS) 
metric to generate positive and negative queries. Additionally, we develop a new classification head 
and a new classification loss that are variations of the LQE head and the varifocal loss used in D-FINE.


## 📝 Updates
- [x] **\[2026.07.10\]** Release DETRPose inference code and 🤗 weights.

## Model Zoo
### COCO val2017
| Model | AP | AP<sup>50</sup> | #Params | Latency | GFLOPs | config |                       checkpoint (HF)                       |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:-----------------------------------------------------------:|
| **DETRPose-N** | 57.2 | 81.7 | 4.1 M | 2.80 ms | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_n.py) | [57.2](https://huggingface.co/SebasJanampa/DETRPose_N_COCO) |
| **DETRPose-S** | 67.0 | 87.6 | 11.5 M | 4.99 ms | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_s.py) | [67.0](https://huggingface.co/SebasJanampa/DETRPose_S_COCO) |
| **DETRPose-M** | 69.4 | 89.2 | 20.8 M | 7.01 ms | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_m.py) | [69.4](https://huggingface.co/SebasJanampa/DETRPose_M_COCO) |
| **DETRPose-L** | 72.5 | 90.6 | 32.8 M | 9.50 ms | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_l.py) | [72.5](https://huggingface.co/SebasJanampa/DETRPose_L_COCO) |
| **DETRPose-X** | 73.3 | 90.5 | 73.3 M | 13.31 ms | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_x.py) | [73.3](https://huggingface.co/SebasJanampa/DETRPose_X_COCO) |

### CrowdPose test
| Model | AP | AP<sup>50</sup> | #Params | Latency | GFLOPs | config |                         checkpoint (HF)                          |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:----------------------------------------------------------------:|
| **DETRPose-N** | 56.0 | 80.7 | 4.1 M | 2.72 ms | 8.8 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_n_crowdpose.py) | [56.0](https://huggingface.co/SebasJanampa/DETRPose_N_CROWDPOSE) |
| **DETRPose-S** | 67.4 | 88.6 | 11.5 M | 4.80 ms | 31.3 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_s_crowdpose.py) | [67.4](https://huggingface.co/SebasJanampa/DETRPose_S_CROWDPOSE) |
| **DETRPose-M** | 72.0 | 91.0 | 20.7 M | 6.86 ms | 64.9 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_m_crowdpose.py) | [72.0](https://huggingface.co/SebasJanampa/DETRPose_M_CROWDPOSE) |
| **DETRPose-L** | 73.3 | 91.6 | 32.7 M | 9.03 ms | 103.5 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_l_crowdpose.py) | [73.3](https://huggingface.co/SebasJanampa/DETRPose_L_CROWDPOSE) |
| **DETRPose-X** | 75.1 | 92.1 | 73.3 M | 13.01 ms | 232.3 | [py](https://github.com/SebastianJanampa/DETRPose/tree/inference_only/detrpose/configs/detrpose/detrpose_hgnetv2_x_crowdpose.py) | [75.1](https://huggingface.co/SebasJanampa/DETRPose_X_CROWDPOSE) |

**Notes:**
- **Latency** is evaluated on a single  Tesla V100 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==8.6.3$.

## 🚀 Installation

You can install the inference-ready version of the library directly from this branch using `pip`:
```shell
pip install git+https://github.com/SebastianJanampa/DETRPose.git@inference_only
````
## 💻 Quick Start
This branch is designed to be incredibly easy to use for inference. 
Here is a quick example of how to load a model and run it on a live webcam feed.
```python
from detrpose import DETR

# Initialization
model = DETR(model='detrpose_hgnetv2_s') # for coco
model = DETR(model='detrpose_hgnetv2_s_crowdpose') # for crowdpose

# Inference
model(source=0) # inference on a webcam
```

## 📚 API Reference

### `DETR` Class Initialization

The core component of the inference library is the `DETR` class.

```python
from detrpose import DETR

model = DETR(model='detrpose_hgnetv2_s', device='cuda', resume=None)
```
### `DETR` Class Initialization Arguments

| Argument | Type | Default | Description | 
| ----- | ----- | ----- | ----- | 
| `model` | `str` | **Required** | The model configuration to use. Can be a built-in string (e.g., `'detrpose_hgnetv2_s'` for COCO, `'detrpose_hgnetv2_s_crowdpose'` for CrowdPose) or a local path to a `.py` config file. | 
| `device` | `str` | `None` | The computing device to use (e.g., `'cuda'`, `'cpu'`, or `'mps'`). If `None`, it auto-detects CUDA availability. | 
| `resume` | `str` | `None` | Path to local `.pth` or `.pt` weights. If `None`, the model automatically pulls the pre-trained deployed weights from the Hugging Face Hub. | 

<br>

### Inference / `__call__` Method Arguments
Calling the instantiated `DETR` model triggers the inference pipeline. It handles dynamic resizing, tensor conversion, forward passes, non-maximum suppression (thresholding), and visualization routing.

| Argument    | Type | Default | Description | 
|-------------| ----- | ----- | ----- | 
| `source`    | `Any` | **Required** | The source data to run inference on (see **Supported Input Sources** below). | 
| `thrh`      | `float` | `0.5` | Confidence threshold. Keypoints with scores below this value are filtered out. | 
| `img_size`  | `int` \| `tuple` | `640` | Target inference size. Automatically rounds dimensions to the nearest multiple of 32 for optimal Transformer processing. | 
| `save_path` | `str` | `None` | Path to save the annotated output. If left as `None` for files/folders, DETRPose auto-generates a filename appending the annotator type (e.g., `output_COCO.jpg`). |


### Supported Input Sources (`source`)

The `DETR` pipeline features a highly robust input handler capable of parsing almost any standard vision format:

1. **Webcam (Live Stream)**
   * **Input:** `0` (Integer) or `"0"` (String).
   * **Behavior:** Opens an interactive OpenCV window showing real-time pose estimation. Press `q` to exit.

2. **Single Image File**
   * **Input:** Filepath string (e.g., `"./test.jpg"`).
   * **Behavior:** Processes the image and auto-saves the annotated image.

3. **Video File**
   * **Input:** Filepath string (e.g., `"./video.mp4"`).
   * **Behavior:** Processes the video frame-by-frame and seamlessly compiles a new annotated `.mp4` file.

4. **Directory / Folder**
   * **Input:** Directory path string (e.g., `"./my_dataset/"`).
   * **Behavior:** Recursively processes all supported images and videos inside the folder, saving the annotated results into an `output/` sub-directory.

5. **Raw Arrays (OpenCV & PIL)**
   * **Input:** `PIL.Image.Image` or `numpy.ndarray` (cv2 BGR format).
   * **Behavior:** Processes the data in memory. Highly useful for integration into larger custom pipelines.


## 📜 Citation
If you use `DETRPose` or its methods in your work, please cite the following BibTeX entries:

```latex
@misc{janampa2025detrpose,
      title={DETRPose: Real-time end-to-end transformer model for multi-person pose estimation}, 
      author={Sebastian Janampa and Marios Pattichis},
      year={2025},
      eprint={2506.13027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.13027}, 
}
```

## 🙏 Acknowledgement
This work was supported in part by [Lambda.ai](https://lambda.ai).

Our work is built upon [DEIM](https://github.com/Intellindust-AI-Lab/DEIM/tree/main), [D-FINE](https://github.com/Peterande/D-FINE), [Detectron2](https://github.com/facebookresearch/detectron2/tree/main), and [GroupPose](https://github.com/Michel-liu/GroupPose/tree/main).

✨ Feel free to contribute and reach out if you have any questions! ✨

<div align="left">
  <a href="https://lambda.ai" target="_blank">
  	<img src="./assets/lambda_logo2.png" width=500 >
  </a>
</div>

