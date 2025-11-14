import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return img


def extract_sift(img):

    try:
        sift = cv2.SIFT_create()
    except Exception:
        raise RuntimeError("SIFT no está disponible. Instala opencv-contrib-python.")

    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def extract_orb(img):

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return descriptors


def extract_features(path, method="sift"):

    img = load_image(path)

    if method == "sift":
        desc = extract_sift(img)
    elif method == "orb":
        desc = extract_orb(img)
    else:
        raise ValueError("Método debe ser 'sift' u 'orb'.")

    if desc is None:
        
        return np.empty((0, 128)) if method == "sift" else np.empty((0,32))

    return desc
