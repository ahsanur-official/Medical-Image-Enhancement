"""
Edge Enhancement Script for Medical Images

Processes images from the img folder using Sobel and Laplacian operators
to enhance visibility and apply edge-preserving enhancements for better
diagnostic quality. Results are saved to enhanced_images folder with
detailed comparison visualizations.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
from medical_image_enhancement import MedicalImageEnhancer


class EdgeEnhancer:
    """
    Class for applying edge-based enhancements to medical images.
    Focuses on Sobel and Laplacian operators for improved visibility.
    """
    
    def __init__(self, input_dir='img', output_dir='enhanced_images'):
        """
        Initialize the edge enhancer.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different enhancement types
        self.sobel_dir = self.output_dir / 'sobel_enhanced'
        self.laplacian_dir = self.output_dir / 'laplacian_enhanced'
        self.comparison_dir = self.output_dir / 'comparisons'
        
        self.sobel_dir.mkdir(exist_ok=True)
        self.laplacian_dir.mkdir(exist_ok=True)
        self.comparison_dir.mkdir(exist_ok=True)
    
    def apply_sobel_enhancement(self, image):
        """
        Apply Sobel edge detection and enhance the original image.
        
        Sobel operator detects gradients/edges and enhances them
        for better visibility of anatomical structures.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Enhanced image with Sobel edge emphasis
        """
        # Apply Sobel in both directions
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        # Enhance the original image with Sobel edges
        # Normalize edges
        edges_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Combine original with edge enhancement
        image_float = image.astype(float)
        enhanced = image_float + (edges_normalized * 255 * 0.5)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced, magnitude
    
    def apply_laplacian_enhancement(self, image):
        """
        Apply Laplacian edge detection and enhance the original image.
        
        Laplacian detects second derivatives (zero-crossings) which are
        excellent for edge detection in medical images.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Enhanced image with Laplacian edge emphasis
        """
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Get absolute value for edge map
        laplacian_abs = np.abs(laplacian)
        laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, 
                                             cv2.NORM_MINMAX).astype(np.uint8)
        
        # Enhance the original image with Laplacian edges
        edges_normalized = laplacian_abs / (np.max(laplacian_abs) + 1e-5)
        
        # Combine original with edge enhancement
        image_float = image.astype(float)
        enhanced = image_float + (edges_normalized * 255 * 0.5)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced, laplacian_normalized
    
    def process_image(self, image_path):
        """
        Process a single image with both Sobel and Laplacian enhancement.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Dictionary with processing results
        """
        image_name = Path(image_path).stem
        
        print(f"\n{'='*70}")
        print(f"Processing: {image_name}")
        print(f"{'='*70}")
        
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"✓ Image loaded: {image.shape[0]}x{image.shape[1]} pixels")
        
        # Get original statistics
        original_stats = self.get_statistics(image)
        
        # Apply Sobel enhancement
        print("\n[1/2] Applying Sobel Edge Enhancement...")
        sobel_enhanced, sobel_edges = self.apply_sobel_enhancement(image)
        sobel_stats = self.get_statistics(sobel_enhanced)
        print("      ✓ Sobel enhancement applied")
        
        # Apply Laplacian enhancement
        print("[2/2] Applying Laplacian Edge Enhancement...")
        laplacian_enhanced, laplacian_edges = self.apply_laplacian_enhancement(image)
        laplacian_stats = self.get_statistics(laplacian_enhanced)
        print("      ✓ Laplacian enhancement applied")
        
        # Save enhanced images
        sobel_path = self.sobel_dir / f"{image_name}_sobel.png"
        laplacian_path = self.laplacian_dir / f"{image_name}_laplacian.png"
        
        cv2.imwrite(str(sobel_path), sobel_enhanced)
        cv2.imwrite(str(laplacian_path), laplacian_enhanced)
        
        print(f"\n✓ Sobel enhanced saved: {sobel_path}")
        print(f"✓ Laplacian enhanced saved: {laplacian_path}")
        
        # Create comparison visualization
        self.create_comparison(image, sobel_enhanced, laplacian_enhanced,
                              sobel_edges, laplacian_edges, image_name)
        
        results = {
            'image_name': image_name,
            'original_image': image,
            'sobel_enhanced': sobel_enhanced,
            'laplacian_enhanced': laplacian_enhanced,
            'sobel_edges': sobel_edges,
            'laplacian_edges': laplacian_edges,
            'original_stats': original_stats,
            'sobel_stats': sobel_stats,
            'laplacian_stats': laplacian_stats
        }
        
        return results
    
    def get_statistics(self, image):
        """
        Calculate image statistics for quality assessment.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with statistics
        """
        return {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'contrast': np.std(image) / np.mean(image) if np.mean(image) > 0 else 0
        }
    
    def create_comparison(self, original, sobel_enhanced, laplacian_enhanced,
                         sobel_edges, laplacian_edges, image_name):
        """
        Create detailed comparison visualization.
        
        Args:
            original: Original image
            sobel_enhanced: Sobel enhanced image
            laplacian_enhanced: Laplacian enhanced image
            sobel_edges: Sobel edge map
            laplacian_edges: Laplacian edge map
            image_name: Name for saving
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Original Image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Original Histogram
        ax2 = plt.subplot(2, 3, 4)
        ax2.hist(original.flatten(), bins=256, color='blue', alpha=0.7)
        ax2.set_title('Original Histogram', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Sobel Enhanced
        ax3 = plt.subplot(2, 3, 2)
        ax3.imshow(sobel_enhanced, cmap='gray')
        ax3.set_title('Sobel Enhanced', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Sobel Histogram
        ax4 = plt.subplot(2, 3, 5)
        ax4.hist(sobel_enhanced.flatten(), bins=256, color='green', alpha=0.7)
        ax4.set_title('Sobel Histogram', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Pixel Intensity')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Laplacian Enhanced
        ax5 = plt.subplot(2, 3, 3)
        ax5.imshow(laplacian_enhanced, cmap='gray')
        ax5.set_title('Laplacian Enhanced', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # Laplacian Histogram
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(laplacian_enhanced.flatten(), bins=256, color='red', alpha=0.7)
        ax6.set_title('Laplacian Histogram', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Pixel Intensity')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_path = self.comparison_dir / f"{image_name}_comparison.png"
        plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
        print(f"✓ Comparison saved: {comparison_path}")
        plt.close()
    
    def create_edge_comparison(self, sobel_edges, laplacian_edges, image_name):
        """
        Create edge detection comparison visualization.
        
        Args:
            sobel_edges: Sobel edge map
            laplacian_edges: Laplacian edge map
            image_name: Name for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(sobel_edges, cmap='hot')
        axes[0].set_title('Sobel Edges', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(laplacian_edges, cmap='hot')
        axes[1].set_title('Laplacian Edges', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        edge_path = self.comparison_dir / f"{image_name}_edges_comparison.png"
        plt.savefig(str(edge_path), dpi=150, bbox_inches='tight')
        print(f"✓ Edge comparison saved: {edge_path}")
        plt.close()
    
    def process_all_images(self):
        """
        Process all images in the input directory.
        
        Returns:
            List of processing results
        """
        results = []
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
            image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("❌ No images found in img folder")
            return results
        
        print(f"\n🔍 Found {len(image_files)} image(s) to process")
        
        for image_file in sorted(image_files):
            result = self.process_image(image_file)
            if result:
                results.append(result)
                self.create_edge_comparison(result['sobel_edges'], 
                                          result['laplacian_edges'],
                                          result['image_name'])
        
        return results
    
    def print_statistics_report(self, results):
        """
        Print detailed statistics report for all processed images.
        
        Args:
            results: List of processing results
        """
        print("\n\n" + "="*90)
        print("ENHANCEMENT STATISTICS REPORT")
        print("="*90)
        
        for result in results:
            image_name = result['image_name']
            
            print(f"\n📊 {image_name.upper()}")
            print("-"*90)
            
            print(f"\n{'Metric':<20}{'Original':>20}{'Sobel Enhanced':>20}{'Laplacian Enhanced':>20}")
            print("-"*90)
            
            stats_keys = ['mean', 'std', 'min', 'max', 'median', 'contrast']
            
            for key in stats_keys:
                orig_val = result['original_stats'][key]
                sobel_val = result['sobel_stats'][key]
                lap_val = result['laplacian_stats'][key]
                
                print(f"{key.upper():<20}{orig_val:>20.2f}{sobel_val:>20.2f}{lap_val:>20.2f}")
            
            # Calculate improvements
            orig_contrast = result['original_stats']['contrast']
            sobel_contrast = result['sobel_stats']['contrast']
            lap_contrast = result['laplacian_stats']['contrast']
            
            sobel_improvement = (
                ((sobel_contrast - orig_contrast) / orig_contrast * 100) 
                if orig_contrast > 0 else 0
            )
            lap_improvement = (
                ((lap_contrast - orig_contrast) / orig_contrast * 100) 
                if orig_contrast > 0 else 0
            )
            
            print(f"\n{'IMPROVEMENTS':<20}")
            print(f"  Sobel Contrast:     {sobel_improvement:>+6.2f}%")
            print(f"  Laplacian Contrast: {lap_improvement:>+6.2f}%")


def main():
    """Main execution function."""
    
    print("\n" + "="*90)
    print("MEDICAL IMAGE ENHANCEMENT WITH EDGE DETECTION")
    print("Sobel and Laplacian Enhancement Pipeline")
    print("="*90)
    
    # Check if img folder exists
    img_dir = Path('img')
    if not img_dir.exists():
        print("❌ img folder not found!")
        return
    
    # Initialize enhancer
    enhancer = EdgeEnhancer(input_dir='img', output_dir='enhanced_images')
    
    print("\n📂 Input Directory: img/")
    print(f"📁 Output Directory: {enhancer.output_dir}/")
    print(f"   ├── sobel_enhanced/")
    print(f"   ├── laplacian_enhanced/")
    print(f"   └── comparisons/")
    
    # Process all images
    results = enhancer.process_all_images()
    
    if not results:
        print("\n❌ No images were processed")
        return
    
    # Print statistics report
    enhancer.print_statistics_report(results)
    
    # Summary
    print("\n\n" + "="*90)
    print("✅ PROCESSING COMPLETE")
    print("="*90)
    print(f"\n📈 Processed {len(results)} image(s)")
    print(f"💾 Enhanced images saved to: enhanced_images/")
    print(f"📊 Comparisons saved to: enhanced_images/comparisons/")
    print("\n📝 Output Structure:")
    print("   enhanced_images/")
    print("   ├── sobel_enhanced/      (Sobel-enhanced medical images)")
    print("   ├── laplacian_enhanced/  (Laplacian-enhanced medical images)")
    print("   └── comparisons/         (Comparison visualizations)")
    print("\n" + "="*90)
    
    return results


if __name__ == "__main__":
    results = main()
