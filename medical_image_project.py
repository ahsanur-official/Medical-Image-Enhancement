"""
Unified Medical Image Enhancement Project

This file merges the full project into a single runnable script with a CLI.
"""

import argparse
from pathlib import Path
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")


class MedicalImageEnhancer:
    """
    A class to enhance medical images for improved diagnostic visibility.

    Attributes:
        image: The original medical image
        enhanced_images: Dictionary storing different enhancement results
    """

    def __init__(self, image_path=None, image_array=None):
        """
        Initialize the enhancer with either an image file or array.

        Args:
            image_path: Path to the medical image file
            image_array: Numpy array of the image
        """
        if image_path:
            self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                raise ValueError(f"Could not load image from {image_path}")
        elif image_array is not None:
            self.image = image_array.astype(np.uint8)
        else:
            self.image = None

        self.enhanced_images = {}
        self.original_image = self.image.copy() if self.image is not None else None

    def create_sample_medical_image(self, image_type="xray"):
        """
        Create a synthetic medical image for demonstration.

        Args:
            image_type: Type of image ("xray", "mri", or "chest")

        Returns:
            Synthetic medical image
        """
        size = 256
        image = np.zeros((size, size), dtype=np.uint8)

        if image_type == "xray":
            cv2.circle(image, (128, 128), 80, 100, -1)
            cv2.rectangle(image, (80, 100), (176, 160), 80, -1)
            cv2.ellipse(image, (128, 140), (50, 70), 0, 0, 360, 120, -1)

        elif image_type == "mri":
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - 128) ** 2 + (j - 128) ** 2)
                    image[i, j] = int(150 * np.exp(-(dist ** 2) / (2 * 40 ** 2)))
            cv2.ellipse(image, (100, 100), (30, 50), 30, 0, 360, 100, -1)

        elif image_type == "chest":
            cv2.ellipse(image, (128, 128), (70, 100), 0, 0, 360, 80, -1)
            cv2.ellipse(image, (100, 100), (30, 40), 0, 0, 360, 110, -1)
            cv2.ellipse(image, (156, 100), (30, 40), 0, 0, 360, 110, -1)

        noise = np.random.normal(0, 15, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

        self.image = image
        self.original_image = image.copy()
        return image

    def apply_noise_reduction(self, method="bilateral"):
        """Apply edge-preserving noise reduction filters."""
        if self.image is None:
            raise ValueError("No image loaded")

        if method == "bilateral":
            denoised = cv2.bilateralFilter(self.image, 11, 75, 75)
        elif method == "morphological":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            denoised = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        elif method == "nlm":
            denoised = cv2.fastNlMeansDenoising(
                self.image,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.enhanced_images["denoised"] = denoised
        return denoised

    def apply_histogram_equalization(self):
        """Apply standard histogram equalization for contrast enhancement."""
        if self.image is None:
            raise ValueError("No image loaded")

        equalized = cv2.equalizeHist(self.image)
        self.enhanced_images["histogram_eq"] = equalized
        return equalized

    def apply_clahe(self, clip_limit=2.0, tile_size=8):
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        if self.image is None:
            raise ValueError("No image loaded")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(self.image)

        self.enhanced_images["clahe"] = enhanced
        return enhanced

    def apply_adaptive_histogram_equalization(self):
        """Apply adaptive histogram equalization using CLAHE."""
        if self.image is None:
            raise ValueError("No image loaded")

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(self.image)

        self.enhanced_images["adaptive_eq"] = enhanced
        return enhanced

    def apply_edge_preservation(self, sigma=1.0):
        """Apply edge-preserving smoothing using Gaussian filtering."""
        if self.image is None:
            raise ValueError("No image loaded")

        smoothed = gaussian_filter(self.image.astype(float), sigma=sigma)
        edges = cv2.Canny(self.image, 50, 150)

        smoothed_uint8 = np.clip(smoothed, 0, 255).astype(np.uint8)
        self.enhanced_images["edge_preserved"] = smoothed_uint8
        self.enhanced_images["edges"] = edges

        return smoothed_uint8, edges

    def detect_edges(self, method="canny"):
        """Detect edges in the image using various methods."""
        if self.image is None:
            raise ValueError("No image loaded")

        if method == "canny":
            edges = cv2.Canny(self.image, 50, 150)
        elif method == "sobel":
            sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx ** 2 + sobely ** 2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        elif method == "laplacian":
            laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
            edges = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.enhanced_images[f"edges_{method}"] = edges
        return edges

    def apply_complete_enhancement_pipeline(self):
        """Apply complete enhancement pipeline with denoising and CLAHE."""
        if self.image is None:
            raise ValueError("No image loaded")

        denoised = self.apply_noise_reduction(method="bilateral")
        self.apply_clahe(clip_limit=2.0, tile_size=8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        final = clahe.apply(denoised)

        self.enhanced_images["pipeline"] = final
        return final

    def get_image_statistics(self, image=None):
        """Calculate and return image statistics for quality assessment."""
        img = image if image is not None else self.image

        if img is None:
            raise ValueError("No image to analyze")

        return {
            "mean": np.mean(img),
            "std": np.std(img),
            "min": np.min(img),
            "max": np.max(img),
            "median": np.median(img),
            "contrast": np.std(img) / np.mean(img) if np.mean(img) > 0 else 0,
        }

    def compare_enhancements(self):
        """Compare original and enhanced images with statistics."""
        results = {
            "original": self.get_image_statistics(self.original_image),
            "enhanced": {},
        }

        for name, enhanced_img in self.enhanced_images.items():
            if enhanced_img is not None:
                results["enhanced"][name] = self.get_image_statistics(enhanced_img)

        return results

    def visualize_enhancements(self, save_path=None):
        """Create a comprehensive visualization of all enhancements."""
        if self.image is None:
            raise ValueError("No image loaded")

        num_images = len(self.enhanced_images) + 1
        num_cols = 3
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten() if num_images > 1 else np.array([axes])

        axes[0].imshow(self.original_image, cmap="gray")
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        for idx, (name, enhanced_img) in enumerate(self.enhanced_images.items(), 1):
            if idx < len(axes) and enhanced_img is not None:
                if len(enhanced_img.shape) == 2:
                    axes[idx].imshow(enhanced_img, cmap="gray")
                else:
                    axes[idx].imshow(enhanced_img)

                axes[idx].set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
                axes[idx].axis("off")

        for idx in range(len(self.enhanced_images) + 1, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.close()

    def visualize_comparison(self, enhanced_key="pipeline", save_path=None):
        """Create a detailed comparison between original and enhanced image."""
        if enhanced_key not in self.enhanced_images:
            raise ValueError(f"Enhancement '{enhanced_key}' not found")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(self.original_image, cmap="gray")
        axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[1, 0].hist(self.original_image.flatten(), bins=256, color="blue", alpha=0.7)
        axes[1, 0].set_title("Original Histogram", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Pixel Value")
        axes[1, 0].set_ylabel("Frequency")

        enhanced = self.enhanced_images[enhanced_key]
        axes[0, 1].imshow(enhanced, cmap="gray")
        axes[0, 1].set_title(f"Enhanced Image ({enhanced_key})", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        axes[1, 1].hist(enhanced.flatten(), bins=256, color="green", alpha=0.7)
        axes[1, 1].set_title("Enhanced Histogram", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Pixel Value")
        axes[1, 1].set_ylabel("Frequency")

        diff = cv2.absdiff(self.original_image, enhanced)
        axes[0, 2].imshow(diff, cmap="hot")
        axes[0, 2].set_title("Difference Map", fontsize=12, fontweight="bold")
        axes[0, 2].axis("off")

        stats_original = self.get_image_statistics(self.original_image)
        stats_enhanced = self.get_image_statistics(enhanced)

        stats_text = "Statistics Comparison:\n\n"
        stats_text += f"{'Metric':<15}{'Original':>15}{'Enhanced':>15}\n"
        stats_text += "-" * 45 + "\n"
        for key in stats_original.keys():
            stats_text += f"{key:<15}{stats_original[key]:>15.2f}{stats_enhanced[key]:>15.2f}\n"

        axes[1, 2].text(
            0.1,
            0.5,
            stats_text,
            fontsize=10,
            family="monospace",
            verticalalignment="center",
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison saved to {save_path}")

        plt.close()


class EdgeEnhancer:
    """
    Class for applying edge-based enhancements to medical images.
    Focuses on Sobel and Laplacian operators for improved visibility.
    """

    def __init__(self, input_dir="img", output_dir="enhanced_images"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.sobel_dir = self.output_dir / "sobel_enhanced"
        self.laplacian_dir = self.output_dir / "laplacian_enhanced"
        self.comparison_dir = self.output_dir / "comparisons"

        self.sobel_dir.mkdir(exist_ok=True)
        self.laplacian_dir.mkdir(exist_ok=True)
        self.comparison_dir.mkdir(exist_ok=True)

    def apply_sobel_enhancement(self, image):
        """Apply Sobel edge detection and enhance the original image."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        edges_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        image_float = image.astype(float)
        enhanced = image_float + (edges_normalized * 255 * 0.5)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced, magnitude

    def apply_laplacian_enhancement(self, image):
        """Apply Laplacian edge detection and enhance the original image."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        laplacian_abs = np.abs(laplacian)
        laplacian_normalized = cv2.normalize(
            laplacian_abs, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        edges_normalized = laplacian_abs / (np.max(laplacian_abs) + 1e-5)

        image_float = image.astype(float)
        enhanced = image_float + (edges_normalized * 255 * 0.5)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced, laplacian_normalized

    def process_image(self, image_path):
        """Process a single image with both Sobel and Laplacian enhancement."""
        image_name = Path(image_path).stem

        print("\n" + "=" * 70)
        print(f"Processing: {image_name}")
        print("=" * 70)

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None

        print(f"OK: Image loaded: {image.shape[0]}x{image.shape[1]} pixels")

        original_stats = self.get_statistics(image)

        print("\n[1/2] Applying Sobel Edge Enhancement...")
        sobel_enhanced, sobel_edges = self.apply_sobel_enhancement(image)
        sobel_stats = self.get_statistics(sobel_enhanced)
        print("      OK: Sobel enhancement applied")

        print("[2/2] Applying Laplacian Edge Enhancement...")
        laplacian_enhanced, laplacian_edges = self.apply_laplacian_enhancement(image)
        laplacian_stats = self.get_statistics(laplacian_enhanced)
        print("      OK: Laplacian enhancement applied")

        sobel_path = self.sobel_dir / f"{image_name}_sobel.png"
        laplacian_path = self.laplacian_dir / f"{image_name}_laplacian.png"

        cv2.imwrite(str(sobel_path), sobel_enhanced)
        cv2.imwrite(str(laplacian_path), laplacian_enhanced)

        print(f"\nOK: Sobel enhanced saved: {sobel_path}")
        print(f"OK: Laplacian enhanced saved: {laplacian_path}")

        self.create_comparison(
            image,
            sobel_enhanced,
            laplacian_enhanced,
            sobel_edges,
            laplacian_edges,
            image_name,
        )

        results = {
            "image_name": image_name,
            "original_image": image,
            "sobel_enhanced": sobel_enhanced,
            "laplacian_enhanced": laplacian_enhanced,
            "sobel_edges": sobel_edges,
            "laplacian_edges": laplacian_edges,
            "original_stats": original_stats,
            "sobel_stats": sobel_stats,
            "laplacian_stats": laplacian_stats,
        }

        return results

    def get_statistics(self, image):
        """Calculate image statistics for quality assessment."""
        return {
            "mean": np.mean(image),
            "std": np.std(image),
            "min": np.min(image),
            "max": np.max(image),
            "median": np.median(image),
            "contrast": np.std(image) / np.mean(image) if np.mean(image) > 0 else 0,
        }

    def create_comparison(
        self,
        original,
        sobel_enhanced,
        laplacian_enhanced,
        sobel_edges,
        laplacian_edges,
        image_name,
    ):
        """Create detailed comparison visualization."""
        fig = plt.figure(figsize=(20, 12))

        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original, cmap="gray")
        ax1.set_title("Original Image", fontsize=14, fontweight="bold")
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, 4)
        ax2.hist(original.flatten(), bins=256, color="blue", alpha=0.7)
        ax2.set_title("Original Histogram", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Pixel Intensity")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 3, 2)
        ax3.imshow(sobel_enhanced, cmap="gray")
        ax3.set_title("Sobel Enhanced", fontsize=14, fontweight="bold")
        ax3.axis("off")

        ax4 = plt.subplot(2, 3, 5)
        ax4.hist(sobel_enhanced.flatten(), bins=256, color="green", alpha=0.7)
        ax4.set_title("Sobel Histogram", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Pixel Intensity")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(2, 3, 3)
        ax5.imshow(laplacian_enhanced, cmap="gray")
        ax5.set_title("Laplacian Enhanced", fontsize=14, fontweight="bold")
        ax5.axis("off")

        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(laplacian_enhanced.flatten(), bins=256, color="red", alpha=0.7)
        ax6.set_title("Laplacian Histogram", fontsize=12, fontweight="bold")
        ax6.set_xlabel("Pixel Intensity")
        ax6.set_ylabel("Frequency")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        comparison_path = self.comparison_dir / f"{image_name}_comparison.png"
        plt.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
        print(f"OK: Comparison saved: {comparison_path}")
        plt.close()

    def create_edge_comparison(self, sobel_edges, laplacian_edges, image_name):
        """Create edge detection comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].imshow(sobel_edges, cmap="hot")
        axes[0].set_title("Sobel Edges", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(laplacian_edges, cmap="hot")
        axes[1].set_title("Laplacian Edges", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()

        edge_path = self.comparison_dir / f"{image_name}_edges_comparison.png"
        plt.savefig(str(edge_path), dpi=150, bbox_inches="tight")
        print(f"OK: Edge comparison saved: {edge_path}")
        plt.close()

    def process_all_images(self):
        """Process all images in the input directory."""
        results = []

        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print("ERROR: No images found in img folder")
            return results

        print(f"\nFound {len(image_files)} image(s) to process")

        for image_file in sorted(image_files):
            result = self.process_image(image_file)
            if result:
                results.append(result)
                self.create_edge_comparison(
                    result["sobel_edges"],
                    result["laplacian_edges"],
                    result["image_name"],
                )

        return results

    def print_statistics_report(self, results):
        """Print detailed statistics report for all processed images."""
        print("\n\n" + "=" * 90)
        print("ENHANCEMENT STATISTICS REPORT")
        print("=" * 90)

        for result in results:
            image_name = result["image_name"]

            print(f"\n{image_name.upper()}")
            print("-" * 90)

            print(
                f"\n{'Metric':<20}{'Original':>20}{'Sobel Enhanced':>20}{'Laplacian Enhanced':>20}"
            )
            print("-" * 90)

            stats_keys = ["mean", "std", "min", "max", "median", "contrast"]

            for key in stats_keys:
                orig_val = result["original_stats"][key]
                sobel_val = result["sobel_stats"][key]
                lap_val = result["laplacian_stats"][key]

                print(f"{key.upper():<20}{orig_val:>20.2f}{sobel_val:>20.2f}{lap_val:>20.2f}")

            orig_contrast = result["original_stats"]["contrast"]
            sobel_contrast = result["sobel_stats"]["contrast"]
            lap_contrast = result["laplacian_stats"]["contrast"]

            sobel_improvement = (
                ((sobel_contrast - orig_contrast) / orig_contrast * 100)
                if orig_contrast > 0
                else 0
            )
            lap_improvement = (
                ((lap_contrast - orig_contrast) / orig_contrast * 100)
                if orig_contrast > 0
                else 0
            )

            print("\nIMPROVEMENTS")
            print(f"  Sobel Contrast:     {sobel_improvement:+.2f}%")
            print(f"  Laplacian Contrast: {lap_improvement:+.2f}%")


def create_comprehensive_visualization(img_dir="img", enhanced_dir="enhanced_images"):
    """Create comprehensive visualization of all enhancements."""
    img_dir = Path(img_dir)
    enhanced_dir = Path(enhanced_dir)

    image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))

    for img_file in sorted(image_files):
        image_name = img_file.stem

        original = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue

        sobel_path = enhanced_dir / "sobel_enhanced" / f"{image_name}_sobel.png"
        laplacian_path = enhanced_dir / "laplacian_enhanced" / f"{image_name}_laplacian.png"

        if not sobel_path.exists() or not laplacian_path.exists():
            continue

        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.imread(str(laplacian_path), cv2.IMREAD_GRAYSCALE)

        fig = plt.figure(figsize=(20, 10))

        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original, cmap="gray")
        ax1.set_title("Original Image", fontsize=14, fontweight="bold", color="darkblue")
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(sobel, cmap="gray")
        ax2.set_title("Sobel Enhanced\n(Edge Gradient)", fontsize=14, fontweight="bold", color="darkgreen")
        ax2.axis("off")

        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(laplacian, cmap="gray")
        ax3.set_title("Laplacian Enhanced\n(Fine Details)", fontsize=14, fontweight="bold", color="darkred")
        ax3.axis("off")

        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(original.flatten(), bins=256, color="blue", alpha=0.6, label="Original")
        ax4.hist(sobel.flatten(), bins=256, color="green", alpha=0.4, label="Sobel")
        ax4.set_title("Histogram Comparison - Sobel vs Original", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Pixel Intensity")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(original.flatten(), bins=256, color="blue", alpha=0.6, label="Original")
        ax5.hist(laplacian.flatten(), bins=256, color="red", alpha=0.4, label="Laplacian")
        ax5.set_title("Histogram Comparison - Laplacian vs Original", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Pixel Intensity")
        ax5.set_ylabel("Frequency")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        orig_stats = {
            "mean": np.mean(original),
            "std": np.std(original),
            "contrast": np.std(original) / np.mean(original) if np.mean(original) > 0 else 0,
            "min": np.min(original),
            "max": np.max(original),
        }

        sobel_stats = {
            "mean": np.mean(sobel),
            "std": np.std(sobel),
            "contrast": np.std(sobel) / np.mean(sobel) if np.mean(sobel) > 0 else 0,
            "min": np.min(sobel),
            "max": np.max(sobel),
        }

        laplacian_stats = {
            "mean": np.mean(laplacian),
            "std": np.std(laplacian),
            "contrast": np.std(laplacian) / np.mean(laplacian) if np.mean(laplacian) > 0 else 0,
            "min": np.min(laplacian),
            "max": np.max(laplacian),
        }

        stats_text = "IMAGE STATISTICS\n" + "=" * 50 + "\n\n"
        stats_text += f"{'Metric':<15} {'Original':>12} {'Sobel':>12} {'Laplacian':>12}\n"
        stats_text += "-" * 52 + "\n"
        stats_text += f"{'Mean':<15} {orig_stats['mean']:>12.2f} {sobel_stats['mean']:>12.2f} {laplacian_stats['mean']:>12.2f}\n"
        stats_text += f"{'Std Dev':<15} {orig_stats['std']:>12.2f} {sobel_stats['std']:>12.2f} {laplacian_stats['std']:>12.2f}\n"
        stats_text += f"{'Contrast':<15} {orig_stats['contrast']:>12.3f} {sobel_stats['contrast']:>12.3f} {laplacian_stats['contrast']:>12.3f}\n"
        stats_text += f"{'Min':<15} {orig_stats['min']:>12.0f} {sobel_stats['min']:>12.0f} {laplacian_stats['min']:>12.0f}\n"
        stats_text += f"{'Max':<15} {orig_stats['max']:>12.0f} {sobel_stats['max']:>12.0f} {laplacian_stats['max']:>12.0f}\n"

        sobel_contrast_improve = (
            ((sobel_stats["contrast"] - orig_stats["contrast"]) / orig_stats["contrast"] * 100)
            if orig_stats["contrast"] > 0
            else 0
        )
        laplacian_contrast_improve = (
            ((laplacian_stats["contrast"] - orig_stats["contrast"]) / orig_stats["contrast"] * 100)
            if orig_stats["contrast"] > 0
            else 0
        )

        stats_text += "\n" + "-" * 52 + "\n"
        stats_text += "Contrast improvement vs Original:\n"
        stats_text += f"  Sobel:     {sobel_contrast_improve:+.2f}%\n"
        stats_text += f"  Laplacian: {laplacian_contrast_improve:+.2f}%"

        ax6.text(
            0.05,
            0.95,
            stats_text,
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            transform=ax6.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(
            f"Comprehensive Edge Enhancement Analysis - {image_name.upper()}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = enhanced_dir / "comparisons" / f"{image_name}_comprehensive.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"OK: Comprehensive visualization saved: {output_path}")
        plt.close()


def create_side_by_side_enhancement_grid(img_dir="img", enhanced_dir="enhanced_images"):
    """Create a side-by-side grid of all enhancements."""
    img_dir = Path(img_dir)
    enhanced_dir = Path(enhanced_dir)

    image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    image_files = sorted(image_files)

    if not image_files:
        print("No images found")
        return

    fig = plt.figure(figsize=(20, 8 * len(image_files)))

    for idx, img_file in enumerate(image_files):
        image_name = img_file.stem

        original = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        sobel_path = enhanced_dir / "sobel_enhanced" / f"{image_name}_sobel.png"
        laplacian_path = enhanced_dir / "laplacian_enhanced" / f"{image_name}_laplacian.png"

        if not sobel_path.exists() or not laplacian_path.exists():
            continue

        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.imread(str(laplacian_path), cv2.IMREAD_GRAYSCALE)

        sobel_diff = cv2.absdiff(original, sobel)
        laplacian_diff = cv2.absdiff(original, laplacian)

        row = idx

        ax = plt.subplot(len(image_files), 5, row * 5 + 1)
        ax.imshow(original, cmap="gray")
        ax.set_title(f"{image_name} - Original", fontweight="bold")
        ax.axis("off")

        ax = plt.subplot(len(image_files), 5, row * 5 + 2)
        ax.imshow(sobel, cmap="gray")
        ax.set_title(f"{image_name} - Sobel", fontweight="bold", color="green")
        ax.axis("off")

        ax = plt.subplot(len(image_files), 5, row * 5 + 3)
        ax.imshow(laplacian, cmap="gray")
        ax.set_title(f"{image_name} - Laplacian", fontweight="bold", color="red")
        ax.axis("off")

        ax = plt.subplot(len(image_files), 5, row * 5 + 4)
        ax.imshow(sobel_diff, cmap="hot")
        ax.set_title(f"{image_name} - Sobel Diff", fontweight="bold")
        ax.axis("off")

        ax = plt.subplot(len(image_files), 5, row * 5 + 5)
        ax.imshow(laplacian_diff, cmap="hot")
        ax.set_title(f"{image_name} - Laplacian Diff", fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Edge Enhancement Grid - Original vs Sobel vs Laplacian vs Differences",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = enhanced_dir / "comparisons" / "complete_enhancement_grid.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"OK: Complete enhancement grid saved: {output_path}")
    plt.close()


def analyze_folder(folder_path):
    """Analyze images in a folder."""
    items = list(Path(folder_path).glob("*"))
    return len([i for i in items if i.is_file() and i.suffix in [".png", ".jpg"]])


def get_image_stats(image_path):
    """Get image statistics."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    return {
        "shape": img.shape,
        "mean": np.mean(img),
        "std": np.std(img),
        "contrast": np.std(img) / np.mean(img) if np.mean(img) > 0 else 0,
    }


def print_summary(base_dir="enhanced_images"):
    """Print summary of all processed images."""
    print("\n" + "=" * 100)
    print("EDGE ENHANCEMENT PROCESSING - COMPLETE SUMMARY")
    print("=" * 100)

    base_dir = Path(base_dir)

    if not base_dir.exists():
        print("ERROR: No enhanced_images directory found")
        return

    sobel_dir = base_dir / "sobel_enhanced"
    laplacian_dir = base_dir / "laplacian_enhanced"
    comparison_dir = base_dir / "comparisons"

    sobel_count = analyze_folder(sobel_dir)
    laplacian_count = analyze_folder(laplacian_dir)
    comparison_count = analyze_folder(comparison_dir)

    print("\nPROCESSING STATISTICS:")
    print(f"   Sobel Enhanced Images:     {sobel_count}")
    print(f"   Laplacian Enhanced Images: {laplacian_count}")
    print(f"   Comparison Visualizations: {comparison_count}")

    print("\nSOBEL ENHANCED FOLDER:")
    for f in sorted(sobel_dir.glob("*.png")):
        stats = get_image_stats(f)
        if stats:
            print(f"   OK: {f.name}")
            print(f"      Size: {stats['shape'][0]}x{stats['shape'][1]}")
            print(f"      Contrast: {stats['contrast']:.3f}")

    print("\nLAPLACIAN ENHANCED FOLDER:")
    for f in sorted(laplacian_dir.glob("*.png")):
        stats = get_image_stats(f)
        if stats:
            print(f"   OK: {f.name}")
            print(f"      Size: {stats['shape'][0]}x{stats['shape'][1]}")
            print(f"      Contrast: {stats['contrast']:.3f}")

    print("\nCOMPARISONS GENERATED:")
    for f in sorted(comparison_dir.glob("*.png")):
        print(f"   OK: {f.name}")

    print("\nENHANCEMENT COMPLETE")
    print("\nOUTPUT LOCATIONS:")
    print(f"   Sobel Enhanced:      {sobel_dir}")
    print(f"   Laplacian Enhanced:  {laplacian_dir}")
    print(f"   Comparisons:         {comparison_dir}")

    print("\n" + "=" * 100)
    print("KEY ENHANCEMENTS APPLIED:")
    print("=" * 100)
    print(
        """
SOBEL EDGE DETECTION:
  - Detects gradient/edge information in both X and Y directions
  - Enhances edges while preserving original image features
  - Better for showing anatomical boundaries
  - Good for detecting rapid intensity changes

LAPLACIAN EDGE DETECTION:
  - Detects second derivatives (zero-crossings)
  - More sensitive to fine details
  - Highlights internal structure transitions
  - Useful for finding peaks and valleys in intensity

EDGE-ENHANCED VISIBILITY:
  - Original images combined with edge maps
  - 50% weight emphasis on detected edges
  - Maintains original anatomical context
  - Enhanced visibility of anatomical structures

COMPARISON VISUALIZATIONS:
  - Side-by-side original vs enhanced comparisons
  - Histogram analysis for each enhancement
  - Edge detection map comparisons
  - Quality metrics displayed

RECOMMENDED USE CASES:
  - Sobel: Use when focused on anatomical boundaries and edges
  - Laplacian: Use when looking for fine structural details
  - Both: Compare to choose the best enhancement for your analysis
"""
    )
    print("=" * 100)


def demonstrate_medical_image_enhancement():
    """Demonstrate the medical image enhancement pipeline."""
    print("=" * 70)
    print("MEDICAL IMAGE ENHANCEMENT FOR DIAGNOSIS SUPPORT")
    print("=" * 70)

    print("\n1. Creating synthetic medical image...")
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type="chest")
    print("   OK: Synthetic chest X-ray created (256x256)")

    print("\n2. Applying enhancement techniques...")

    print("   a) Noise Reduction (Bilateral Filter)...")
    enhancer.apply_noise_reduction(method="bilateral")
    print("      OK: Bilateral filter applied - preserves edges while reducing noise")

    print("   b) Histogram Equalization...")
    enhancer.apply_histogram_equalization()
    print("      OK: Standard histogram equalization applied")

    print("   c) CLAHE (Contrast Limited Adaptive Histogram Equalization)...")
    enhancer.apply_clahe(clip_limit=2.0, tile_size=8)
    print("      OK: CLAHE applied - prevents noise amplification")

    print("   d) Adaptive Histogram Equalization...")
    enhancer.apply_adaptive_histogram_equalization()
    print("      OK: Adaptive equalization applied")

    print("   e) Edge Preservation...")
    enhancer.apply_edge_preservation(sigma=1.0)
    print("      OK: Edge-preserving smoothing applied")

    print("   f) Edge Detection (Canny)...")
    enhancer.detect_edges(method="canny")
    print("      OK: Canny edge detection applied")

    print("   g) Complete Pipeline (Denoise + CLAHE)...")
    enhancer.apply_complete_enhancement_pipeline()
    print("      OK: Complete enhancement pipeline applied")

    print("\n3. Image Quality Analysis...")
    stats = enhancer.compare_enhancements()

    print("\n   Original Image Statistics:")
    for key, value in stats["original"].items():
        print(f"      {key.capitalize():<15}: {value:>10.2f}")

    print("\n   Enhanced Images (showing pipeline results):")
    if "pipeline" in stats["enhanced"]:
        for key, value in stats["enhanced"]["pipeline"].items():
            print(f"      {key.capitalize():<15}: {value:>10.2f}")

    original_contrast = stats["original"]["contrast"]
    enhanced_contrast = stats["enhanced"]["pipeline"]["contrast"]
    improvement = (
        ((enhanced_contrast - original_contrast) / original_contrast * 100)
        if original_contrast > 0
        else 0
    )

    print(f"\n   Contrast Improvement: {improvement:>6.2f}%")

    print("\n4. Generating visualizations...")
    output_dir = Path("medical_image_output")
    output_dir.mkdir(exist_ok=True)

    comparison_path = output_dir / "enhancement_comparison.png"
    enhancer.visualize_comparison(enhanced_key="pipeline", save_path=str(comparison_path))
    print(f"   OK: Comparison saved to {comparison_path}")

    all_enhancements_path = output_dir / "all_enhancements.png"
    enhancer.visualize_enhancements(save_path=str(all_enhancements_path))
    print(f"   OK: All enhancements saved to {all_enhancements_path}")

    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print("\nKey Learning Outcomes Demonstrated:")
    print("  - Noise reduction with edge-preserving filters (Bilateral Filter)")
    print("  - Contrast enhancement (Histogram Equalization and CLAHE)")
    print("  - Anatomical structure visibility improvement")
    print("  - Image quality assessment and comparison")
    print("  - Edge detection for structure identification")
    print("=" * 70)


def example_1_basic_enhancement():
    """Demonstrates basic image enhancement with a synthetic chest X-ray."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Enhancement of Synthetic Image")
    print("=" * 70)

    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type="chest")

    enhanced = enhancer.apply_complete_enhancement_pipeline()

    print("\nOriginal vs Enhanced Statistics:")
    stats = enhancer.compare_enhancements()

    print(f"Contrast Ratio - Original: {stats['original']['contrast']:.3f}")
    print(f"Contrast Ratio - Enhanced: {stats['enhanced']['pipeline']['contrast']:.3f}")

    enhancer.visualize_comparison(enhanced_key="pipeline", save_path="example1_comparison.png")
    print("\nOK: Example 1 complete - See example1_comparison.png")


def example_2_compare_denoising():
    """Compare different noise reduction techniques."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparing Denoising Methods")
    print("=" * 70)

    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type="xray")

    methods = ["bilateral", "morphological", "nlm"]

    print("\nDenoising Methods Comparison:")
    print(f"{'Method':<20} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 56)

    stats_orig = enhancer.get_image_statistics()
    print(
        f"{'Original':<20} {stats_orig['mean']:>12.2f} "
        f"{stats_orig['std']:>12.2f} {stats_orig['contrast']:>12.3f}"
    )

    for method in methods:
        denoised = enhancer.apply_noise_reduction(method=method)
        stats = enhancer.get_image_statistics(denoised)
        print(
            f"{method.capitalize():<20} {stats['mean']:>12.2f} "
            f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}"
        )

    print("\nOK: Example 2 complete")


def example_3_clahe_tuning():
    """Demonstrate the effect of CLAHE parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CLAHE Parameter Tuning")
    print("=" * 70)

    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type="mri")

    print("\nEffect of clipLimit (tile size = 8x8):")
    print(f"{'ClipLimit':<15} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 51)

    clip_limits = [1.0, 2.0, 3.0, 4.0, 5.0]

    for clip in clip_limits:
        enhanced = enhancer.apply_clahe(clip_limit=clip, tile_size=8)
        stats = enhancer.get_image_statistics(enhanced)
        print(
            f"{clip:<15.1f} {stats['mean']:>12.2f} "
            f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}"
        )

    print("\nEffect of tileGridSize (clipLimit = 2.0):")
    print(f"{'TileSize':<15} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 51)

    tile_sizes = [4, 8, 16, 32]

    for tile in tile_sizes:
        enhanced = enhancer.apply_clahe(clip_limit=2.0, tile_size=tile)
        stats = enhancer.get_image_statistics(enhanced)
        print(
            f"{tile}x{tile:<12} {stats['mean']:>12.2f} "
            f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}"
        )

    print("\nOK: Example 3 complete")
    print("\nRecommendation:")
    print("  - For subtle structures: Use lower clipLimit (1.5-2.0)")
    print("  - For pronounced structures: Use higher clipLimit (3.0-4.0)")
    print("  - Smaller tiles: More local adaptation (better for varied anatomy)")
    print("  - Larger tiles: More global adaptation (smoother result)")


def example_4_edge_detection():
    """Compare different edge detection methods."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Edge Detection Methods")
    print("=" * 70)

    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type="xray")

    enhancer.apply_complete_enhancement_pipeline()
    enhanced = enhancer.enhanced_images["pipeline"]

    methods = ["canny", "sobel", "laplacian"]

    print("\nEdge Detection Statistics:")
    print(f"{'Method':<15} {'Edge Pixels':>15} {'Percentage':>12}")
    print("-" * 45)

    total_pixels = enhanced.shape[0] * enhanced.shape[1]

    for method in methods:
        edges = enhancer.detect_edges(method=method)
        edge_pixels = np.count_nonzero(edges)
        percentage = (edge_pixels / total_pixels) * 100

        print(f"{method.capitalize():<15} {edge_pixels:>15} {percentage:>11.2f}%")

    print("\nOK: Example 4 complete")
    print("\nMethod Characteristics:")
    print("  - Canny: Most selective, best for clear boundaries")
    print("  - Sobel: Detects gradient magnitude, good for varying edges")
    print("  - Laplacian: Detects zero-crossings, sensitive to noise")


def example_5_complete_analysis():
    """Perform complete analysis on all synthetic image types."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Analysis - All Image Types")
    print("=" * 70)

    image_types = ["xray", "mri", "chest"]

    for img_type in image_types:
        print(f"\n{img_type.upper()} Analysis:")
        print("-" * 50)

        enhancer = MedicalImageEnhancer()
        enhancer.create_sample_medical_image(image_type=img_type)

        enhancer.apply_noise_reduction(method="bilateral")
        enhancer.apply_histogram_equalization()
        enhancer.apply_clahe(clip_limit=2.0, tile_size=8)
        enhancer.apply_adaptive_histogram_equalization()
        enhancer.apply_edge_preservation(sigma=1.0)
        enhancer.apply_complete_enhancement_pipeline()

        stats = enhancer.compare_enhancements()

        print("\nOriginal Image:")
        print(f"  Mean:        {stats['original']['mean']:>10.2f}")
        print(f"  Std Dev:     {stats['original']['std']:>10.2f}")
        print(f"  Contrast:    {stats['original']['contrast']:>10.3f}")
        print(
            f"  Min-Max:     {stats['original']['min']:>10.0f} - {stats['original']['max']:<.0f}"
        )

        print("\nBest Enhancement (Pipeline):")
        pipeline_stats = stats["enhanced"]["pipeline"]
        print(f"  Mean:        {pipeline_stats['mean']:>10.2f}")
        print(f"  Std Dev:     {pipeline_stats['std']:>10.2f}")
        print(f"  Contrast:    {pipeline_stats['contrast']:>10.3f}")
        print(f"  Min-Max:     {pipeline_stats['min']:>10.0f} - {pipeline_stats['max']:<.0f}")

        contrast_improvement = (
            (pipeline_stats["contrast"] - stats["original"]["contrast"]) / stats["original"]["contrast"] * 100
        ) if stats["original"]["contrast"] > 0 else 0

        print("\nImprovement:")
        print(f"  Contrast:    {contrast_improvement:>10.2f}%")

    print("\nOK: Example 5 complete")


def example_6_custom_enhancement():
    """Demonstrate how to use the enhancer with custom numpy arrays."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Image Enhancement")
    print("=" * 70)

    custom_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    enhancer = MedicalImageEnhancer(image_array=custom_image)

    enhancer.apply_complete_enhancement_pipeline()
    stats = enhancer.compare_enhancements()

    print(
        f"Original - Mean: {stats['original']['mean']:.2f}, "
        f"Contrast: {stats['original']['contrast']:.3f}"
    )
    print(
        f"Enhanced - Mean: {stats['enhanced']['pipeline']['mean']:.2f}, "
        f"Contrast: {stats['enhanced']['pipeline']['contrast']:.3f}"
    )

    print("\nOK: Example 6 complete")


def example_7_batch_processing():
    """Demonstrate batch processing of multiple images."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Batch Processing")
    print("=" * 70)

    image_types = ["xray", "mri", "chest"]

    print("\nBatch Processing Results:")
    print(f"{'Image Type':<15} {'Original Contrast':>20} {'Enhanced Contrast':>20}")
    print("-" * 57)

    for img_type in image_types:
        enhancer = MedicalImageEnhancer()
        enhancer.create_sample_medical_image(image_type=img_type)
        enhancer.apply_complete_enhancement_pipeline()

        stats = enhancer.compare_enhancements()
        orig_contrast = stats["original"]["contrast"]
        enh_contrast = stats["enhanced"]["pipeline"]["contrast"]

        print(f"{img_type:<15} {orig_contrast:>20.3f} {enh_contrast:>20.3f}")

    print("\nOK: Example 7 complete - Batch processing successful")


def run_examples():
    """Run all examples in order."""
    print("\n" + "*" * 70)
    print("MEDICAL IMAGE ENHANCEMENT - PRACTICAL EXAMPLES")
    print("*" * 70)

    try:
        example_1_basic_enhancement()
        example_2_compare_denoising()
        example_3_clahe_tuning()
        example_4_edge_detection()
        example_5_complete_analysis()
        example_6_custom_enhancement()
        example_7_batch_processing()

        print("\n" + "*" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("*" * 70 + "\n")

    except Exception as exc:
        print(f"\nERROR running examples: {exc}")
        import traceback

        traceback.print_exc()


def run_edge_pipeline(input_dir="img", output_dir="enhanced_images"):
    """Run the Sobel/Laplacian enhancement pipeline on all images."""
    print("\n" + "=" * 90)
    print("MEDICAL IMAGE ENHANCEMENT WITH EDGE DETECTION")
    print("Sobel and Laplacian Enhancement Pipeline")
    print("=" * 90)

    img_dir = Path(input_dir)
    if not img_dir.exists():
        print("ERROR: img folder not found")
        return []

    enhancer = EdgeEnhancer(input_dir=input_dir, output_dir=output_dir)

    print(f"\nInput Directory: {img_dir}/")
    print(f"Output Directory: {enhancer.output_dir}/")
    print("   - sobel_enhanced/")
    print("   - laplacian_enhanced/")
    print("   - comparisons/")

    results = enhancer.process_all_images()

    if not results:
        print("\nERROR: No images were processed")
        return []

    enhancer.print_statistics_report(results)

    print("\n\n" + "=" * 90)
    print("PROCESSING COMPLETE")
    print("=" * 90)
    print(f"\nProcessed {len(results)} image(s)")
    print(f"Enhanced images saved to: {output_dir}/")
    print(f"Comparisons saved to: {output_dir}/comparisons/")
    print("\nOutput Structure:")
    print(f"   {output_dir}/")
    print("   - sobel_enhanced/      (Sobel-enhanced medical images)")
    print("   - laplacian_enhanced/  (Laplacian-enhanced medical images)")
    print("   - comparisons/         (Comparison visualizations)")
    print("\n" + "=" * 90)

    return results


def run_pipeline_on_folder(
    input_dir="img",
    output_dir="enhanced_images",
    show_output=True,
    display_delay_ms=0,
):
    """Run the full enhancement pipeline on all images in a folder."""
    img_dir = Path(input_dir)
    if not img_dir.exists():
        print("ERROR: img folder not found")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    overview_dir = output_dir / "comparisons"
    overview_dir.mkdir(exist_ok=True)

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(img_dir.glob(f"*{ext}"))
        image_files.extend(img_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print("ERROR: No images found in img folder")
        return []

    print(f"\nFound {len(image_files)} image(s) to process")
    print(f"Input Directory: {img_dir}/")
    print(f"Output Directory: {output_dir}/")

    results = []
    overview_items = []
    for image_file in sorted(image_files):
        try:
            enhancer = MedicalImageEnhancer(image_path=image_file)
            enhanced = enhancer.apply_complete_enhancement_pipeline()

            sobel_enhanced = _compute_sobel_enhanced(enhancer.original_image)
            laplacian_enhanced = _compute_laplacian_enhanced(enhancer.original_image)

            output_path = output_dir / f"{image_file.stem}_enhanced.png"
            cv2.imwrite(str(output_path), enhanced)
            results.append(output_path)
            print(f"OK: Saved {output_path}")

            overview_items.append(
                {
                    "name": image_file.stem,
                    "original": enhancer.original_image,
                    "enhanced": enhanced,
                    "sobel": sobel_enhanced,
                    "laplacian": laplacian_enhanced,
                }
            )
        except Exception as exc:
            print(f"ERROR: Failed processing {image_file}: {exc}")

    if show_output and overview_items:
        overview_path = _create_all_images_overview(overview_items, overview_dir)
        overview_image = cv2.imread(str(overview_path))
        if overview_image is not None:
            cv2.imshow("All Images Overview", overview_image)
            cv2.waitKey(display_delay_ms)
            cv2.destroyAllWindows()

    return results


def _compute_sobel_enhanced(image):
    """Compute Sobel-enhanced image for overview display."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    edges_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    image_float = image.astype(float)
    enhanced = image_float + (edges_normalized * 255 * 0.5)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def _compute_laplacian_enhanced(image):
    """Compute Laplacian-enhanced image for overview display."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    edges_normalized = laplacian_abs / (np.max(laplacian_abs) + 1e-5)

    image_float = image.astype(float)
    enhanced = image_float + (edges_normalized * 255 * 0.5)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def _create_pipeline_overview(
    original,
    enhanced,
    sobel_enhanced,
    laplacian_enhanced,
    image_name,
    overview_dir,
):
    """Create a fixed overview image for each input file."""
    fig = plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(enhanced, cmap="gray")
    ax2.set_title("Pipeline Enhanced", fontsize=12, fontweight="bold")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(sobel_enhanced, cmap="gray")
    ax3.set_title("Sobel Enhanced", fontsize=12, fontweight="bold")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(laplacian_enhanced, cmap="gray")
    ax4.set_title("Laplacian Enhanced", fontsize=12, fontweight="bold")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(original.flatten(), bins=256, color="blue", alpha=0.7)
    ax5.set_title("Original Histogram", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Pixel Intensity")
    ax5.set_ylabel("Frequency")

    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(enhanced.flatten(), bins=256, color="green", alpha=0.7)
    ax6.set_title("Enhanced Histogram", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Pixel Intensity")
    ax6.set_ylabel("Frequency")

    plt.suptitle(f"Pipeline Overview - {image_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    overview_path = overview_dir / f"{image_name}_pipeline_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches="tight")
    plt.close()

    return overview_path


def _create_all_images_overview(overview_items, overview_dir):
    """Create a single overview frame for all images."""
    num_images = len(overview_items)
    fig_width = 24
    fig_height = max(6, 5 * num_images)
    fig, axes = plt.subplots(num_images, 4, figsize=(fig_width, fig_height))
    if num_images == 1:
        axes = np.array([axes])

    for row_idx, item in enumerate(overview_items):
        ax = axes[row_idx, 0]
        ax.imshow(item["original"], cmap="gray")
        ax.set_title(f"{item['name']} - Original", fontsize=10, fontweight="bold")
        ax.axis("off")

        ax = axes[row_idx, 1]
        ax.imshow(item["enhanced"], cmap="gray")
        ax.set_title("Pipeline Enhanced", fontsize=10, fontweight="bold")
        ax.axis("off")

        ax = axes[row_idx, 2]
        ax.imshow(item["sobel"], cmap="gray")
        ax.set_title("Sobel Enhanced", fontsize=10, fontweight="bold")
        ax.axis("off")

        ax = axes[row_idx, 3]
        ax.imshow(item["laplacian"], cmap="gray")
        ax.set_title("Laplacian Enhanced", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("All Images Overview", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    overview_path = overview_dir / "all_images_overview.png"
    plt.savefig(str(overview_path), dpi=200, bbox_inches="tight")
    plt.close()

    return overview_path


def build_cli():
    """Build the command line interface."""
    parser = argparse.ArgumentParser(
        description="Unified medical image enhancement project",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    edge_parser = subparsers.add_parser("edge", help="Run Sobel/Laplacian enhancement")
    edge_parser.add_argument("--input-dir", default="img", help="Input image folder")
    edge_parser.add_argument("--output-dir", default="enhanced_images", help="Output folder")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full enhancement pipeline on images in a folder",
    )
    pipeline_parser.add_argument("--input-dir", default="img", help="Input image folder")
    pipeline_parser.add_argument("--output-dir", default="enhanced_images", help="Output folder")
    pipeline_parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show_output",
        help="Disable image display",
    )
    pipeline_parser.add_argument(
        "--display-delay-ms",
        type=int,
        default=0,
        help="Delay per image in milliseconds (0 waits for keypress)",
    )

    viz_parser = subparsers.add_parser("visualize", help="Create comprehensive visualizations")
    viz_parser.add_argument("--input-dir", default="img", help="Input image folder")
    viz_parser.add_argument("--output-dir", default="enhanced_images", help="Output folder")
    viz_parser.add_argument(
        "--mode",
        choices=["comprehensive", "grid", "both"],
        default="both",
        help="Visualization type",
    )

    summary_parser = subparsers.add_parser("summary", help="Print enhancement summary")
    summary_parser.add_argument("--output-dir", default="enhanced_images", help="Output folder")

    subparsers.add_parser("demo", help="Run medical image enhancement demo")
    subparsers.add_parser("examples", help="Run all examples")

    return parser


def main():
    """CLI entry point."""
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "edge":
        run_edge_pipeline(input_dir=args.input_dir, output_dir=args.output_dir)
    elif args.command == "pipeline":
        run_pipeline_on_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            show_output=getattr(args, "show_output", True),
            display_delay_ms=getattr(args, "display_delay_ms", 0),
        )
    elif args.command == "visualize":
        if args.mode in ("comprehensive", "both"):
            create_comprehensive_visualization(args.input_dir, args.output_dir)
        if args.mode in ("grid", "both"):
            create_side_by_side_enhancement_grid(args.input_dir, args.output_dir)
    elif args.command == "summary":
        print_summary(args.output_dir)
    elif args.command == "demo":
        demonstrate_medical_image_enhancement()
    elif args.command == "examples":
        run_examples()


if __name__ == "__main__":
    main()
