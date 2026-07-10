import copy
import torch
import numpy as np


OKS_SIGMAS = {
    'coco': np.array(
        [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]
    ) / 10.0,
    'crowdpose': np.array(
        [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]
    ) / 10.0,
}


def get_oks_sigmas(dataset_type, num_keypoints, oks_sigmas=None):
    dataset_type = dataset_type.lower()

    if oks_sigmas is not None and dataset_type not in ["coco", "crowdpose"]:
        oks_sigmas = np.array(oks_sigmas)
    elif dataset_type in ["coco", "crowdpose"]:
        oks_sigmas = copy.deepcopy(OKS_SIGMAS)[dataset_type]
    elif num_keypoints == 17:
        oks_sigmas = copy.deepcopy(OKS_SIGMAS)["coco"]
    elif num_keypoints == 14:
        oks_sigmas = copy.deepcopy(OKS_SIGMAS)["crowdpose"]
    else:
        oks_sigmas = np.array([1.0 / num_keypoints] * num_keypoints)

    return oks_sigmas


def keypoint_xyxyzz_to_xyzxyz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    Z = keypoints[..., :2*num_points]
    V = keypoints[..., 2*num_points:]
    res[...,0::3] = Z[..., 0::2]
    res[...,1::3] = Z[..., 1::2]
    res[...,2::3] = V[...]
    return res

def keypoint_xyzxyz_to_xyxyzz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    res[...,0:2*num_points:2] = keypoints[..., 0::3]
    res[...,1:2*num_points:2] = keypoints[..., 1::3]
    res[...,2*num_points:] = keypoints[..., 2::3]
    return res