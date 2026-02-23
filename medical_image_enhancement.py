"""
Medical Image Enhancement Pipeline for Diagnosis Support

This module implements a comprehensive image enhancement pipeline for medical images
such as X-rays and MRI scans. It includes:
- Noise reduction using edge-preserving filters
- Histogram equalization for contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Anatomical structure visibility improvement
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')


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
    
    def create_sample_medical_image(self, image_type='xray'):
        """
        Create a synthetic medical image for demonstration.
        
        Args:
            image_type: Type of image ('xray', 'mri', or 'chest')
        
        Returns:
            Synthetic medical image
        """
        size = 256
        image = np.zeros((size, size), dtype=np.uint8)
        
        if image_type == 'xray':
            # Create synthetic X-ray-like image with anatomical structures
            cv2.circle(image, (128, 128), 80, 100, -1)  # Lung area
            cv2.rectangle(image, (80, 100), (176, 160), 80, -1)  # Heart area
            cv2.ellipse(image, (128, 140), (50, 70), 0, 0, 360, 120, -1)
            
        elif image_type == 'mri':
            # Create synthetic MRI-like image
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - 128)**2 + (j - 128)**2)
                    image[i, j] = int(150 * np.exp(-(dist**2) / (2 * 40**2)))
            # Add some structures
            cv2.ellipse(image, (100, 100), (30, 50), 30, 0, 360, 100, -1)
            
        elif image_type == 'chest':
            # Create synthetic chest X-ray
            cv2.ellipse(image, (128, 128), (70, 100), 0, 0, 360, 80, -1)
            cv2.ellipse(image, (100, 100), (30, 40), 0, 0, 360, 110, -1)
            cv2.ellipse(image, (156, 100), (30, 40), 0, 0, 360, 110, -1)
        
        # Add noise to make it more realistic
        noise = np.random.normal(0, 15, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        self.image = image
        self.original_image = image.copy()
        return image
    
    def apply_noise_reduction(self, method='bilateral'):
        """
        Apply edge-preserving noise reduction filters.
        
        Args:
            method: 'bilateral', 'morphological', or 'nlm'
        
        Returns:
            Denoised image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        if method == 'bilateral':
            # Bilateral filtering: reduces noise while preserving edges
            denoised = cv2.bilateralFilter(self.image, 11, 75, 75)
            
        elif method == 'morphological':
            # Morphological closing then opening for noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            denoised = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            
        elif method == 'nlm':
            # Non-local means denoising (slower but very effective)
            denoised = cv2.fastNlMeansDenoising(
                self.image, 
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.enhanced_images['denoised'] = denoised
        return denoised
    
    def apply_histogram_equalization(self):
        """
        Apply standard histogram equalization for contrast enhancement.
        
        Returns:
            Histogram equalized image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Standard histogram equalization
        equalized = cv2.equalizeHist(self.image)
        self.enhanced_images['histogram_eq'] = equalized
        return equalized
    
    def apply_clahe(self, clip_limit=2.0, tile_size=8):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        CLAHE improves local contrast while preventing over-amplification of noise.
        
        Args:
            clip_limit: Threshold for contrast limiting (typically 2.0-4.0)
            tile_size: Size of grid tiles for local histogram (typically 4-16)
        
        Returns:
            CLAHE enhanced image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(self.image)
        
        self.enhanced_images['clahe'] = enhanced
        return enhanced
    
    def apply_adaptive_histogram_equalization(self):
        """
        Apply adaptive histogram equalization using CV2 CLAHE.
        
        Returns:
            Adaptively equalized image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Use CLAHE for adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(self.image)
        
        self.enhanced_images['adaptive_eq'] = enhanced
        return enhanced
    
    def apply_edge_preservation(self, sigma=1.0):
        """
        Apply edge-preserving smoothing using Gaussian filtering.
        
        Args:
            sigma: Standard deviation for Gaussian kernel
        
        Returns:
            Edge-preserved smoothed image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(self.image.astype(float), sigma=sigma)
        
        # Calculate edge map (high-pass filter)
        edges = cv2.Canny(self.image, 50, 150)
        
        # Create edge-preserved enhanced image by combining smoothing with edges
        smoothed_uint8 = np.clip(smoothed, 0, 255).astype(np.uint8)
        self.enhanced_images['edge_preserved'] = smoothed_uint8
        self.enhanced_images['edges'] = edges
        
        return smoothed_uint8, edges
    
    def detect_edges(self, method='canny'):
        """
        Detect edges in the image using various methods.
        
        Args:
            method: 'canny', 'sobel', or 'laplacian'
        
        Returns:
            Edge-detected image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        if method == 'canny':
            edges = cv2.Canny(self.image, 50, 150)
            
        elif method == 'sobel':
            # Sobel edge detection (detects gradients)
            sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
            
        elif method == 'laplacian':
            # Laplacian edge detection (detects second derivative)
            laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
            edges = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.enhanced_images[f'edges_{method}'] = edges
        return edges
    
    def apply_complete_enhancement_pipeline(self):
        """
        Apply complete enhancement pipeline with denoising and CLAHE.
        
        Returns:
            Final enhanced image
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Step 1: Denoise using bilateral filter
        denoised = self.apply_noise_reduction(method='bilateral')
        
        # Step 2: Apply CLAHE for contrast enhancement
        enhanced = self.apply_clahe(clip_limit=2.0, tile_size=8)
        
        # Use denoised image for CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        final = clahe.apply(denoised)
        
        self.enhanced_images['pipeline'] = final
        return final
    
    def get_image_statistics(self, image=None):
        """
        Calculate and return image statistics for quality assessment.
        
        Args:
            image: Image to analyze (uses original if None)
        
        Returns:
            Dictionary containing image statistics
        """
        img = image if image is not None else self.image
        
        if img is None:
            raise ValueError("No image to analyze")
        
        stats = {
            'mean': np.mean(img),
            'std': np.std(img),
            'min': np.min(img),
            'max': np.max(img),
            'median': np.median(img),
            'contrast': np.std(img) / np.mean(img) if np.mean(img) > 0 else 0
        }
        
        return stats
    
    def compare_enhancements(self):
        """
        Compare original and enhanced images with statistics.
        
        Returns:
            Dictionary with comparison results
        """
        results = {
            'original': self.get_image_statistics(self.original_image),
            'enhanced': {}
        }
        
        for name, enhanced_img in self.enhanced_images.items():
            if enhanced_img is not None:
                results['enhanced'][name] = self.get_image_statistics(enhanced_img)
        
        return results
    
    def visualize_enhancements(self, save_path=None):
        """
        Create a comprehensive visualization of all enhancements.
        
        Args:
            save_path: Path to save the comparison figure
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Determine number of images to display
        num_images = len(self.enhanced_images) + 1  # +1 for original
        
        # Create layout
        num_cols = 3
        num_rows = (num_images + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten() if num_images > 1 else np.array([axes])
        
        # Plot original image
        axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot enhanced images
        for idx, (name, enhanced_img) in enumerate(self.enhanced_images.items(), 1):
            if idx < len(axes) and enhanced_img is not None:
                if len(enhanced_img.shape) == 2:  # Grayscale
                    axes[idx].imshow(enhanced_img, cmap='gray')
                else:  # Color image (unlikely for medical images)
                    axes[idx].imshow(enhanced_img)
                
                axes[idx].set_title(name.replace('_', ' ').title(), 
                                   fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        # Hide remaining axes
        for idx in range(len(self.enhanced_images) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()
    
    def visualize_comparison(self, enhanced_key='pipeline', save_path=None):
        """
        Create a detailed comparison between original and enhanced image.
        
        Args:
            enhanced_key: Key of the enhanced image to compare
            save_path: Path to save the comparison figure
        """
        if enhanced_key not in self.enhanced_images:
            raise ValueError(f"Enhancement '{enhanced_key}' not found")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image and histogram
        axes[0, 0].imshow(self.original_image, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].hist(self.original_image.flatten(), bins=256, color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Enhanced image and histogram
        enhanced = self.enhanced_images[enhanced_key]
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title(f'Enhanced Image ({enhanced_key})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 1].hist(enhanced.flatten(), bins=256, color='green', alpha=0.7)
        axes[1, 1].set_title('Enhanced Histogram', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Difference image
        diff = cv2.absdiff(self.original_image, enhanced)
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title('Difference Map', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Statistics comparison
        stats_original = self.get_image_statistics(self.original_image)
        stats_enhanced = self.get_image_statistics(enhanced)
        
        stats_text = "Statistics Comparison:\n\n"
        stats_text += f"{'Metric':<15}{'Original':>15}{'Enhanced':>15}\n"
        stats_text += "-" * 45 + "\n"
        for key in stats_original.keys():
            stats_text += f"{key:<15}{stats_original[key]:>15.2f}{stats_enhanced[key]:>15.2f}\n"
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.close()


def demonstrate_medical_image_enhancement():
    """
    Demonstrate the medical image enhancement pipeline with various techniques.
    """
    print("=" * 70)
    print("MEDICAL IMAGE ENHANCEMENT FOR DIAGNOSIS SUPPORT")
    print("=" * 70)
    
    # Create enhancer and synthetic image
    print("\n1. Creating synthetic medical image...")
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type='chest')
    print("   ✓ Synthetic chest X-ray created (256x256)")
    
    # Apply various enhancement techniques
    print("\n2. Applying enhancement techniques...")
    
    print("   a) Noise Reduction (Bilateral Filter)...")
    enhancer.apply_noise_reduction(method='bilateral')
    print("      ✓ Bilateral filter applied - preserves edges while reducing noise")
    
    print("   b) Histogram Equalization...")
    enhancer.apply_histogram_equalization()
    print("      ✓ Standard histogram equalization applied")
    
    print("   c) CLAHE (Contrast Limited Adaptive Histogram Equalization)...")
    enhancer.apply_clahe(clip_limit=2.0, tile_size=8)
    print("      ✓ CLAHE applied - prevents noise amplification")
    
    print("   d) Adaptive Histogram Equalization...")
    enhancer.apply_adaptive_histogram_equalization()
    print("      ✓ Adaptive equalization applied")
    
    print("   e) Edge Preservation...")
    enhancer.apply_edge_preservation(sigma=1.0)
    print("      ✓ Edge-preserving smoothing applied")
    
    print("   f) Edge Detection (Canny)...")
    enhancer.detect_edges(method='canny')
    print("      ✓ Canny edge detection applied")
    
    print("   g) Complete Pipeline (Denoise + CLAHE)...")
    enhancer.apply_complete_enhancement_pipeline()
    print("      ✓ Complete enhancement pipeline applied")
    
    # Display statistics
    print("\n3. Image Quality Analysis...")
    stats = enhancer.compare_enhancements()
    
    print("\n   Original Image Statistics:")
    for key, value in stats['original'].items():
        print(f"      {key.capitalize():<15}: {value:>10.2f}")
    
    print("\n   Enhanced Images (showing pipline results):")
    if 'pipeline' in stats['enhanced']:
        for key, value in stats['enhanced']['pipeline'].items():
            print(f"      {key.capitalize():<15}: {value:>10.2f}")
    
    # Calculate improvement
    original_contrast = stats['original']['contrast']
    enhanced_contrast = stats['enhanced']['pipeline']['contrast']
    improvement = ((enhanced_contrast - original_contrast) / original_contrast * 100) if original_contrast > 0 else 0
    
    print(f"\n   Contrast Improvement: {improvement:>6.2f}%")
    
    # Create visualizations
    print("\n4. Generating visualizations...")
    
    # Save to output directory
    output_dir = Path('medical_image_output')
    output_dir.mkdir(exist_ok=True)
    
    comparison_path = output_dir / 'enhancement_comparison.png'
    enhancer.visualize_comparison(enhanced_key='pipeline', save_path=str(comparison_path))
    print(f"   ✓ Comparison saved to {comparison_path}")
    
    all_enhancements_path = output_dir / 'all_enhancements.png'
    enhancer.visualize_enhancements(save_path=str(all_enhancements_path))
    print(f"   ✓ All enhancements saved to {all_enhancements_path}")
    
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print("\nKey Learning Outcomes Demonstrated:")
    print("  ✓ Noise reduction with edge-preserving filters (Bilateral Filter)")
    print("  ✓ Contrast enhancement (Histogram Equalization & CLAHE)")
    print("  ✓ Anatomical structure visibility improvement")
    print("  ✓ Image quality assessment and comparison")
    print("  ✓ Edge detection for structure identification")
    print("=" * 70)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_medical_image_enhancement()
    
    # Example: Using the enhancer with a custom image
    # enhancer = MedicalImageEnhancer(image_path='path_to_your_medical_image.jpg')
    # enhanced = enhancer.apply_complete_enhancement_pipeline()
    # enhancer.visualize_comparison(enhanced_key='pipeline')
