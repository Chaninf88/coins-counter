"""Coin counting utilities with no third-party dependencies."""

from __future__ import annotations

import io
import math
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # Optional Pillow support for real image IO and Streamlit previews.
    from PIL import Image, ImageOps  # type: ignore
except Exception:  # pragma: no cover - Pillow not available in minimal environments
    Image = None  # type: ignore
    ImageOps = None  # type: ignore


Color = Tuple[int, int, int]


def _clamp(value: float, lower: int = 0, upper: int = 255) -> int:
    return max(lower, min(upper, int(round(value))))


@dataclass
class SimpleImage:
    width: int
    height: int
    pixels: List[List[Color]]

    @classmethod
    def new(cls, width: int, height: int, color: Color = (0, 0, 0)) -> "SimpleImage":
        row = [color for _ in range(width)]
        pixels = [list(row) for _ in range(height)]
        return cls(width, height, pixels)

    def copy(self) -> "SimpleImage":
        return SimpleImage(self.width, self.height, [list(row) for row in self.pixels])

    def get(self, x: int, y: int) -> Color:
        return self.pixels[y][x]

    def set(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def to_bytes(self) -> bytes:
        # Encode as a simple PPM image for environments without Pillow.
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        body = bytearray()
        for row in self.pixels:
            for r, g, b in row:
                body.extend(bytes((_clamp(r), _clamp(g), _clamp(b))))
        return header + bytes(body)


def _from_pillow(img: "Image.Image") -> SimpleImage:
    img = img.convert("RGB")
    width, height = img.size
    data = list(img.getdata())
    pixels: List[List[Color]] = []
    for y in range(height):
        start = y * width
        row = [tuple(map(int, data[start + x])) for x in range(width)]
        pixels.append(row)
    return SimpleImage(width, height, pixels)


def _to_pillow(img: SimpleImage) -> "Image.Image":  # pragma: no cover - requires Pillow
    assert Image is not None
    pil_img = Image.new("RGB", (img.width, img.height))
    flat = [pixel for row in img.pixels for pixel in row]
    pil_img.putdata(flat)
    return pil_img


def load_image(image_bytes: bytes) -> SimpleImage:
    if Image is None:
        raise RuntimeError("Image loading requires Pillow, which is not installed.")
    with Image.open(io.BytesIO(image_bytes)) as pil_img:
        if ImageOps is not None:
            pil_img = ImageOps.exif_transpose(pil_img)
        return _from_pillow(pil_img)


def resize_max(image: SimpleImage, max_size: int = 1200) -> Tuple[SimpleImage, float]:
    longest = max(image.width, image.height)
    if longest <= max_size:
        return image.copy(), 1.0
    scale = max_size / float(longest)
    new_width = max(1, int(image.width * scale))
    new_height = max(1, int(image.height * scale))
    resized = SimpleImage.new(new_width, new_height)
    for y in range(new_height):
        src_y = int(y / scale)
        for x in range(new_width):
            src_x = int(x / scale)
            resized.pixels[y][x] = image.pixels[src_y][src_x]
    return resized, scale


def _grayscale(image: SimpleImage) -> List[List[int]]:
    gray: List[List[int]] = []
    for row in image.pixels:
        gray_row = [_clamp(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in row]
        gray.append(gray_row)
    return gray


def _box_blur(gray: List[List[int]]) -> List[List[int]]:
    height = len(gray)
    width = len(gray[0]) if height else 0
    blurred = [[0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            total = 0
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        total += gray[nr][nc]
                        count += 1
            blurred[r][c] = total // count if count else gray[r][c]
    return blurred


def enhance_contrast_gray(image: SimpleImage) -> List[List[int]]:
    return _box_blur(_grayscale(image))


def _otsu_threshold(gray: Sequence[Sequence[int]]) -> int:
    histogram = [0] * 256
    for row in gray:
        for value in row:
            histogram[value] += 1
    total = sum(histogram)
    sum_total = sum(i * hist for i, hist in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    max_variance = 0.0
    for intensity, hist in enumerate(histogram):
        weight_background += hist
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += intensity * hist
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = intensity
    return best_threshold


def _build_mask(gray: Sequence[Sequence[int]], threshold: int) -> List[List[bool]]:
    return [[value > threshold for value in row] for row in gray]


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[Tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                yield nr, nc


def _connected_components(mask: Sequence[Sequence[bool]]) -> List[List[Tuple[int, int]]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    components: List[List[Tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r][c] or visited[r][c]:
                continue
            stack = [(r, c)]
            visited[r][c] = True
            component: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, height, width):
                    if mask[nr][nc] and not visited[nr][nc]:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            components.append(component)
    return components


def _component_perimeter(component: Sequence[Tuple[int, int]], mask: Sequence[Sequence[bool]]) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    perimeter = 0
    for r, c in component:
        neighbours = 0
        if r > 0 and mask[r - 1][c]:
            neighbours += 1
        if r + 1 < height and mask[r + 1][c]:
            neighbours += 1
        if c > 0 and mask[r][c - 1]:
            neighbours += 1
        if c + 1 < width and mask[r][c + 1]:
            neighbours += 1
        perimeter += 4 - neighbours
    return perimeter


def detect_coins(image: SimpleImage,
                 min_area: int,
                 max_area: int,
                 circularity_threshold: float) -> List[Tuple[int, int, int]]:
    gray = enhance_contrast_gray(image)
    threshold = _otsu_threshold(gray)
    mask = _build_mask(gray, threshold)
    components = _connected_components(mask)
    coins: List[Tuple[int, int, int]] = []
    for component in components:
        area = len(component)
        if area < max(min_area, 5):
            continue
        if max_area > 0 and area > max_area:
            continue
        perimeter = _component_perimeter(component, mask)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / float(perimeter * perimeter)
        if circularity < circularity_threshold:
            continue
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        center_y = int(round(sum(rows) / len(rows)))
        center_x = int(round(sum(cols) / len(cols)))
        radius = int(round(math.sqrt(area / math.pi)))
        coins.append((center_x, center_y, max(radius, 1)))
    return coins


def draw_annotations(image: SimpleImage, circles: Sequence[Tuple[int, int, int]]) -> SimpleImage:
    annotated = image.copy()
    for x, y, radius in circles:
        for angle in range(0, 360):
            rad = math.radians(angle)
            px = int(round(x + radius * math.cos(rad)))
            py = int(round(y + radius * math.sin(rad)))
            annotated.set(px, py, (0, 255, 0))
        annotated.set(x, y, (255, 0, 0))
    return annotated


def count_coins(image: SimpleImage,
                min_area: int = 400,
                max_area: int = 0,
                circularity_threshold: float = 0.60) -> List[Tuple[int, int, int]]:
    return detect_coins(image, min_area=min_area, max_area=max_area, circularity_threshold=circularity_threshold)


def streamlit_app() -> None:  # pragma: no cover - UI glue
    import streamlit as st

    st.set_page_config(page_title="Coin Counter", page_icon="🪙")
    st.title("🪙 Coin Counter")
    st.write("Upload a photo of coins and the app will estimate how many are present.")

    with st.sidebar:
        st.header("Detection settings")
        min_area = st.slider("Minimum coin area", 50, 2000, 400, 10)
        max_area = st.slider("Maximum coin area (0 = auto)", 0, 10000, 0, 50)
        circularity_threshold = st.slider("Minimum circularity", 0.4, 1.0, 0.60, 0.01)

    uploaded = st.file_uploader("Upload a coin image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded is None:
        st.info("Choose an image file to start counting coins.")
        return

    data = uploaded.getvalue()
    if not data:
        st.warning("The uploaded file is empty. Please try again with a valid image.")
        return

    try:
        image = load_image(data)
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"Unable to load image: {exc}")
        return

    resized, scale = resize_max(image, 1200)
    st.write(f"Image resized with scale factor {scale:.2f} for processing.")

    circles = count_coins(resized, min_area=min_area, max_area=max_area, circularity_threshold=circularity_threshold)
    annotated = draw_annotations(resized, circles)

    st.success(f"Detected {len(circles)} coin{'s' if len(circles) != 1 else ''}.")

    if Image is not None:
        st.image(_to_pillow(resized), caption="Processed image")
        st.image(_to_pillow(annotated), caption="Detected coins")
    else:  # pragma: no cover - when Pillow unavailable
        st.download_button("Download annotated image", data=annotated.to_bytes(), file_name="annotated.ppm")


def _draw_circle(image: SimpleImage, center: Tuple[int, int], radius: int) -> None:
    x0, y0 = center
    for y in range(image.height):
        for x in range(image.width):
            if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                image.set(x, y, (255, 255, 255))


def _run_cli(input_path: str, output_path: Optional[str]) -> int:
    with open(input_path, "rb") as fh:
        data = fh.read()
    image = load_image(data)
    resized, _ = resize_max(image, 1200)
    circles = count_coins(resized)
    annotated = draw_annotations(resized, circles)
    if output_path:
        if Image is None:
            with open(output_path, "wb") as out_fh:
                out_fh.write(annotated.to_bytes())
        else:
            _to_pillow(annotated).save(output_path)
    print(len(circles))
    return 0


def _run_tests() -> int:
    import unittest

    class CoinCounterTests(unittest.TestCase):
        def test_three_circles(self) -> None:
            image = SimpleImage.new(400, 400)
            _draw_circle(image, (100, 100), 30)
            _draw_circle(image, (300, 120), 35)
            _draw_circle(image, (220, 300), 25)
            coins = count_coins(image, min_area=200)
            self.assertGreaterEqual(len(coins), 3)

        def test_resize_max(self) -> None:
            image = SimpleImage.new(2000, 1000)
            resized, scale = resize_max(image, 1200)
            self.assertLessEqual(max(resized.width, resized.height), 1200)
            self.assertLessEqual(scale, 1.0)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoinCounterTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output annotated image path")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args, _ = parser.parse_known_args()

    if args.test:
        return _run_tests()

    if args.input:
        return _run_cli(args.input, args.output)

    try:
        import importlib
        importlib.import_module("streamlit")
    except Exception:
        print("Streamlit is not installed. Use '--input <path>' for CLI or '--test' to run tests.")
        return 0

    streamlit_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
