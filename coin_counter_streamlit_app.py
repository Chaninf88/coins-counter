"""
Simple Coin Counter App (Web/CLI) without hard dependency on Streamlit
"""

import io
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    print("[DEBUG] Loading image to BGR...")
    bio = io.BytesIO(image_bytes)
    pil_img = Image.open(bio)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = np.array(pil_img)
    print(f"[DEBUG] Image shape (RGB): {img.shape}")
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_max(img: np.ndarray, max_size: int = 1200) -> Tuple[np.ndarray, float]:
    print(f"[DEBUG] Resizing image, original shape: {img.shape}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    print(f"[DEBUG] Resized shape: {img.shape}, scale: {scale}")
    return img, scale


def enhance_contrast_gray(gray: np.ndarray) -> np.ndarray:
    print("[DEBUG] Enhancing contrast (CLAHE)...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_coins_hough(bgr: np.ndarray, dp: float, min_dist: float, canny: int,
                        acc_threshold: int, min_radius: int, max_radius: int) -> List[Tuple[int, int, int]]:
    print("[DEBUG] Detecting coins using HoughCircles...")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    gray = enhance_contrast_gray(gray)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=float(min_dist),
        param1=int(canny),
        param2=int(acc_threshold),
        minRadius=int(min_radius),
        maxRadius=int(max_radius) if max_radius > 0 else 0,
    )
    if circles is None:
        print("[DEBUG] No circles found with HoughCircles.")
        return []
    circles = np.uint16(np.around(circles[0]))
    print(f"[DEBUG] HoughCircles detected {len(circles)} circles.")
    return [(int(x), int(y), int(r)) for x, y, r in circles]


def detect_coins_contours(bgr: np.ndarray, min_area: int, max_area: int, circularity_threshold: float) -> List[Tuple[int, int, int]]:
    print("[DEBUG] Detecting coins using contours...")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast_gray(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 5)
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Found {len(cnts)} contours.")
    coins = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(min_area, 5):
            continue
        if max_area > 0 and area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < circularity_threshold:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        coins.append((int(x), int(y), int(r)))
    print(f"[DEBUG] Contour-based detection found {len(coins)} circles.")
    return coins


def draw_annotations(bgr: np.ndarray, circles: List[Tuple[int, int, int]]) -> np.ndarray:
    print(f"[DEBUG] Drawing annotations for {len(circles)} circles.")
    out = bgr.copy()
    for (x, y, r) in circles:
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), 2)
    cv2.putText(out, f"Coins: {len(circles)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(out, f"Coins: {len(circles)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return out


def count_coins(bgr: np.ndarray,
                dp: float = 1.2,
                min_dist: float = 40,
                canny: int = 120,
                acc_threshold: int = 35,
                min_radius: int = 15,
                max_radius: int = 120,
                min_area: int = 400,
                max_area: int = 0,
                circularity_threshold: float = 0.70) -> List[Tuple[int, int, int]]:
    print("[DEBUG] Counting coins...")
    circles = detect_coins_hough(
        bgr, dp=dp, min_dist=min_dist, canny=canny,
        acc_threshold=acc_threshold, min_radius=min_radius, max_radius=max_radius
    )
    if len(circles) == 0:
        print("[DEBUG] Falling back to contour detection.")
        circles = detect_coins_contours(
            bgr, min_area=int(min_area), max_area=int(max_area), circularity_threshold=float(circularity_threshold)
        )
    print(f"[DEBUG] Total coins detected: {len(circles)}")
    return circles


def _run_cli(input_path: str, output_path: Optional[str]) -> int:
    print(f"[DEBUG] Running CLI mode with input: {input_path}")
    with open(input_path, "rb") as f:
        b = f.read()
    bgr = load_image_to_bgr(b)
    bgr_resized, _ = resize_max(bgr, 1200)
    circles = count_coins(bgr_resized)
    annotated = draw_annotations(bgr_resized, circles)
    if output_path:
        print(f"[DEBUG] Saving annotated image to: {output_path}")
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(output_path)
    print(f"[DEBUG] Coins counted: {len(circles)}")
    print(len(circles))
    return 0


def _run_tests() -> int:
    print("[DEBUG] Running unit tests...")
    import unittest

    class CoinCounterTests(unittest.TestCase):
        def test_three_circles_contour(self):
            print("[DEBUG] Running test_three_circles_contour")
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.circle(img, (100, 100), 30, (255, 255, 255), -1)
            cv2.circle(img, (300, 120), 35, (255, 255, 255), -1)
            cv2.circle(img, (220, 300), 25, (255, 255, 255), -1)
            found = count_coins(img, min_area=200)
            self.assertGreaterEqual(len(found), 3)

        def test_resize_max(self):
            print("[DEBUG] Running test_resize_max")
            img = np.zeros((2000, 1000, 3), dtype=np.uint8)
            resized, scale = resize_max(img, 1200)
            self.assertTrue(max(resized.shape[:2]) <= 1200)
            self.assertLessEqual(scale, 1.0)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoinCounterTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"[DEBUG] Tests completed. Success: {result.wasSuccessful()}")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    print("[DEBUG] Starting main()...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output annotated image path")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args, _ = parser.parse_known_args()

    print(f"[DEBUG] Parsed args: {args}")

    if args.test:
        return _run_tests()

    if args.input:
        return _run_cli(args.input, args.output)

    try:
        import importlib
        importlib.import_module("streamlit")
    except Exception:
        print("[DEBUG] Streamlit not found.")
        print("Streamlit is not installed. Use '--input <path>' for CLI or '--test' to run tests.")
        return 0

    print("[DEBUG] Launching Streamlit app...")
    from streamlit import runtime
    streamlit_app()
    return 0


if __name__ == "__main__":
    main()
