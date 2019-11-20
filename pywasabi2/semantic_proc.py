"""Set of primitives to process semantic maps and colors."""
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

import pywasabi2.cst as cst

def col2lab(col, colors=cst.palette_bgr):
    """Convert color map to label map.
    WARNING: use cst.palette_bgr from the cityscapes palette and assumes the colors
    are ordered to match their label.

    Args:
        col: (h,w,3) color semantic img.
        colors: list of bgr values ordered in the label order.

    Returns:
        lab: (h,w) label semantic img.
    """
    lab = cst.LABEL_IGNORE * np.ones(col.shape[:2]).astype(np.uint8)
    for i, color in enumerate(colors):
        # I know, this is ugly 
        mask = np.zeros(col.shape[:2]).astype(np.uint8)
        mask = 255*(col==color).astype(np.uint8)
        mask = (np.sum(mask,axis=2) == (255*3)).astype(np.uint8)
        lab[mask==1] = i
    return lab


def lab2col(lab, colors=cst.palette_bgr):
    """Convert label map to color map.
    
    Args:
        lab: (h,w) label semantic img.
        colors: list of bgr values ordered in the label order.

    Returns:
        col: (h,w,3) color semantic img.
    """
    col = np.zeros((lab.shape + (3,))).astype(np.uint8)
    labels = np.unique(lab)
    if np.max(labels) >= len(colors):
        raise ValueError("Error: you need more colors np.max(labels) >= "
                "len(colors): %d >= %d"%(np.max(labels), len(colors)) )
    for label in labels:
        col[lab==label,:] = colors[label]
    return col
