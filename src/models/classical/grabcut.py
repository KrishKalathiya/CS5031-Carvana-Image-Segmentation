import cv2
import numpy as np


def create_seed_mask(scores, erosion_size=5):
    """Mark high-confidence interiors as definite, leaving the rest probable."""
    seeds = np.where(scores >= 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
    for confident, label in ((scores <= 0.05, cv2.GC_BGD), (scores >= 0.95, cv2.GC_FGD)):
        interior = cv2.erode(confident.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        seeds[interior != 0] = label
    return seeds


def foreground(classes):
    return (classes == cv2.GC_FGD) | (classes == cv2.GC_PR_FGD)


def refine_grabcut(image, forest_mask, scores, iterations=5, erosion_size=5):
    """Return a binary mask, the initial seed map, and an optional fallback reason."""
    seeds = create_seed_mask(scores, erosion_size)
    count = np.count_nonzero(foreground(seeds))
    if min(count, seeds.size - count) < 50:
        return forest_mask, seeds, "too few foreground or background pixels"

    classes = seeds.copy()
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros_like(background_model)
    try:
        cv2.setRNGSeed(42)
        cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), classes, None,
                    background_model, foreground_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        return forest_mask, seeds, f"OpenCV error ({error.code})"
    return foreground(classes).astype(np.uint8), seeds, ""
