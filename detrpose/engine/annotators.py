from .base_annotator import BasePoseVisualizer


class COCOVisualizer(BasePoseVisualizer):
    """Configuration for COCO 17-keypoint format."""

    HEX_COLORS = {
        'head': '#1B00FF', 'torso': '#E203FF',
        'lower_body': '#36FF2B', 'connector': '#FF8000'
    }

    HEAD_KPTS = {0, 1, 2, 3, 4}
    LOWER_BODY_KPTS = {13, 14, 15, 16}
    HEAD_TO_TORSO = {(3, 5), (4, 6)}

    def _define_skeleton(self):
        return [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (0, 1), (0, 2), (1, 3), (2, 4), (5, 11),
            (6, 12), (3, 5), (4, 6)
        ]

    def _assign_keypoint_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}

        for i in range(17):
            if i in self.HEAD_KPTS:
                c = bgr['head']
            elif i in self.LOWER_BODY_KPTS:
                c = bgr['lower_body']
            else:
                c = bgr['torso']
            colors.append(c)
        return colors

    def _assign_limb_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}
        connector_limbs = [tuple(sorted(x)) for x in self.HEAD_TO_TORSO]

        for p1, p2 in self.skeleton:
            pair = tuple(sorted((p1, p2)))
            if pair in connector_limbs:
                c = bgr['connector']
            elif p1 in self.HEAD_KPTS and p2 in self.HEAD_KPTS:
                c = bgr['head']
            elif p1 in self.LOWER_BODY_KPTS and p2 in self.LOWER_BODY_KPTS:
                c = bgr['lower_body']
            else:
                c = bgr['torso']
            colors.append(c)
        return colors


class CrowdPoseVisualizer(BasePoseVisualizer):
    """
    Configuration for CrowdPose 14-keypoint format.
    Simplified coloring: Head, Torso (inc. arms), Lower Body, Connectors.
    """

    # Use the same palette as COCO
    HEX_COLORS = {
        'head': '#1B00FF',  # Blue
        'torso': '#E203FF',  # Magenta
        'lower_body': '#36FF2B',  # Green
        'connector': '#FF8000'  # Orange
    }

    # --- Group Definitions ---
    # 12: Top Head, 13: Neck
    HEAD_KPTS = {12, 13}

    # 8,9: Knees, 10,11: Ankles
    LOWER_BODY_KPTS = {8, 9, 10, 11}

    # Limbs that connect Head/Neck to the Shoulders
    # (Neck -> L_Shoulder), (Neck -> R_Shoulder)
    HEAD_TO_TORSO_LIMBS = {(13, 0), (13, 1)}

    def _define_skeleton(self):
        return [
            (12, 13),  # Head-Neck
            (13, 0), (13, 1),  # Neck-Shoulder (Connectors)
            (0, 2), (2, 4),  # Left Arm
            (1, 3), (3, 5),  # Right Arm
            (0, 6), (1, 7),  # Torso Vertical
            (6, 7),  # Hips
            (6, 8), (8, 10),  # Left Leg
            (7, 9), (9, 11)  # Right Leg
        ]

    def _assign_keypoint_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}

        for i in range(14):
            if i in self.HEAD_KPTS:
                c = bgr['head']
            elif i in self.LOWER_BODY_KPTS:
                c = bgr['lower_body']
            else:
                # Includes Shoulders (0,1), Elbows (2,3), Wrists (4,5), Hips (6,7)
                c = bgr['torso']
            colors.append(c)
        return colors

    def _assign_limb_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}

        # Pre-sort connector tuples for easy lookup
        connector_limbs = [tuple(sorted(x)) for x in self.HEAD_TO_TORSO_LIMBS]

        for p1, p2 in self.skeleton:
            pair = tuple(sorted((p1, p2)))

            if pair in connector_limbs:
                c = bgr['connector']
            elif p1 in self.HEAD_KPTS and p2 in self.HEAD_KPTS:
                c = bgr['head']
            elif p1 in self.LOWER_BODY_KPTS and p2 in self.LOWER_BODY_KPTS:
                c = bgr['lower_body']
            else:
                # Everything else (Arms, Torso, Hips) is Torso color
                c = bgr['torso']
            colors.append(c)
        return colors