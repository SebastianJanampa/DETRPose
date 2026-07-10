import numpy as np
import cv2
from typing import List, Tuple


class BasePoseVisualizer:
    """
    Base class for visualizing skeleton keypoints with dynamic scaling.
    """

    def __init__(self, base_thickness=2, canvas_ref_length=640):
        """
        Args:
            base_thickness: Line thickness at reference resolution.
            base_radius: Circle radius at reference resolution.
            canvas_ref_length: The reference dimension (width or height)
                               used for scaling.
        """
        self.base_thickness = base_thickness
        self.ref_len = canvas_ref_length

        # Child classes must populate these
        self.skeleton: List[Tuple[int, int]] = self._define_skeleton()
        self.kpt_colors: List[Tuple[int, int, int]] = self._assign_keypoint_colors()
        self.limb_colors: List[Tuple[int, int, int]] = self._assign_limb_colors()

        self.num_kpts = len(self.kpt_colors)

    # --- Abstract Methods ---
    def _define_skeleton(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def _assign_keypoint_colors(self) -> List[Tuple[int, int, int]]:
        raise NotImplementedError

    def _assign_limb_colors(self) -> List[Tuple[int, int, int]]:
        raise NotImplementedError

    # --- Utilities ---
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

    def draw_on(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draws skeletons on the image in-place.
        Args:
            image: OpenCV image (H, W, 3).
            keypoints: Numpy array of shape (N, K, 2) in absolute pixel coordinates.
        """
        h, w = image.shape[:2]

        # --- Dynamic Scaling Logic ---
        # Calculate scale based on the largest image dimension
        scale_factor = max(w, h) / self.ref_len

        # Scale and ensure minimum size of 1 pixel
        radius = max(2, int(self.base_thickness * scale_factor))
        thickness = radius // 2

        # Ensure integer coordinates
        keypoints = keypoints.astype(int)

        for person_kpts in keypoints:
            visibility = self._get_visibility(person_kpts, w, h)
            # Pass the dynamic sizes to the drawing helper
            self._draw_person(image, person_kpts, visibility, thickness, radius)

        return image

    def _get_visibility(self, coords: np.ndarray, w: int, h: int) -> np.ndarray:
        # Check 1: Not (0,0)
        is_nonzero = np.sum(coords, axis=1) > 0
        # Check 2: Inside bounds
        x, y = coords[:, 0], coords[:, 1]
        is_inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return is_nonzero & is_inside

    def _draw_person(self, img, coords, vis, thickness, radius):
        # 1. Draw Limbs
        for i, (p1, p2) in enumerate(self.skeleton):
            if p1 >= self.num_kpts or p2 >= self.num_kpts: continue

            if vis[p1] and vis[p2]:
                cv2.line(img, tuple(coords[p1]), tuple(coords[p2]),
                         self.limb_colors[i], thickness, cv2.LINE_AA)

        # 2. Draw Joints
        for i in range(self.num_kpts):
            if vis[i]:
                cv2.circle(img, tuple(coords[i]), radius,
                           self.kpt_colors[i], -1, cv2.LINE_AA)