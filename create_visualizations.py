"""
Comprehensive Edge Enhancement Visualization
Creates a complete overview of all Sobel and Laplacian enhancements
for quick comparison and analysis
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def create_comprehensive_visualization():
    """Create comprehensive visualization of all enhancements."""
    
    img_dir = Path('img')
    enhanced_dir = Path('enhanced_images')
    
    # Get all image files from img folder
    image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    
    for img_file in sorted(image_files):
        image_name = img_file.stem
        
        # Load original
        original = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue
        
        # Load enhanced versions
        sobel_path = enhanced_dir / 'sobel_enhanced' / f'{image_name}_sobel.png'
        laplacian_path = enhanced_dir / 'laplacian_enhanced' / f'{image_name}_laplacian.png'
        
        if not sobel_path.exists() or not laplacian_path.exists():
            continue
        
        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.imread(str(laplacian_path), cv2.IMREAD_GRAYSCALE)
        
        # Create comprehensive comparison
        fig = plt.figure(figsize=(20, 10))
        
        # Row 1: Images
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold', color='darkblue')
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(sobel, cmap='gray')
        ax2.set_title('Sobel Enhanced\n(Edge Gradient)', fontsize=14, fontweight='bold', color='darkgreen')
        ax2.axis('off')
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(laplacian, cmap='gray')
        ax3.set_title('Laplacian Enhanced\n(Fine Details)', fontsize=14, fontweight='bold', color='darkred')
        ax3.axis('off')
        
        # Row 2: Histograms
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(original.flatten(), bins=256, color='blue', alpha=0.6, label='Original')
        ax4.hist(sobel.flatten(), bins=256, color='green', alpha=0.4, label='Sobel')
        ax4.set_title('Histogram Comparison - Sobel vs Original', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Pixel Intensity')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(original.flatten(), bins=256, color='blue', alpha=0.6, label='Original')
        ax5.hist(laplacian.flatten(), bins=256, color='red', alpha=0.4, label='Laplacian')
        ax5.set_title('Histogram Comparison - Laplacian vs Original', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Pixel Intensity')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        orig_stats = {
            'mean': np.mean(original),
            'std': np.std(original),
            'contrast': np.std(original) / np.mean(original) if np.mean(original) > 0 else 0,
            'min': np.min(original),
            'max': np.max(original)
        }
        
        sobel_stats = {
            'mean': np.mean(sobel),
            'std': np.std(sobel),
            'contrast': np.std(sobel) / np.mean(sobel) if np.mean(sobel) > 0 else 0,
            'min': np.min(sobel),
            'max': np.max(sobel)
        }
        
        laplacian_stats = {
            'mean': np.mean(laplacian),
            'std': np.std(laplacian),
            'contrast': np.std(laplacian) / np.mean(laplacian) if np.mean(laplacian) > 0 else 0,
            'min': np.min(laplacian),
            'max': np.max(laplacian)
        }
        
        stats_text = "IMAGE STATISTICS\n" + "="*50 + "\n\n"
        stats_text += f"{'Metric':<15} {'Original':>12} {'Sobel':>12} {'Laplacian':>12}\n"
        stats_text += "-"*52 + "\n"
        stats_text += f"{'Mean':<15} {orig_stats['mean']:>12.2f} {sobel_stats['mean']:>12.2f} {laplacian_stats['mean']:>12.2f}\n"
        stats_text += f"{'Std Dev':<15} {orig_stats['std']:>12.2f} {sobel_stats['std']:>12.2f} {laplacian_stats['std']:>12.2f}\n"
        stats_text += f"{'Contrast':<15} {orig_stats['contrast']:>12.3f} {sobel_stats['contrast']:>12.3f} {laplacian_stats['contrast']:>12.3f}\n"
        stats_text += f"{'Min':<15} {orig_stats['min']:>12.0f} {sobel_stats['min']:>12.0f} {laplacian_stats['min']:>12.0f}\n"
        stats_text += f"{'Max':<15} {orig_stats['max']:>12.0f} {sobel_stats['max']:>12.0f} {laplacian_stats['max']:>12.0f}\n"
        
        sobel_contrast_improve = (
            ((sobel_stats['contrast'] - orig_stats['contrast']) / orig_stats['contrast'] * 100)
            if orig_stats['contrast'] > 0 else 0
        )
        laplacian_contrast_improve = (
            ((laplacian_stats['contrast'] - orig_stats['contrast']) / orig_stats['contrast'] * 100)
            if orig_stats['contrast'] > 0 else 0
        )
        
        stats_text += "\n" + "-"*52 + "\n"
        stats_text += f"Contrast improvement vs Original:\n"
        stats_text += f"  Sobel:     {sobel_contrast_improve:+.2f}%\n"
        stats_text += f"  Laplacian: {laplacian_contrast_improve:+.2f}%"
        
        ax6.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Comprehensive Edge Enhancement Analysis - {image_name.upper()}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = enhanced_dir / 'comparisons' / f'{image_name}_comprehensive.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"✓ Comprehensive visualization saved: {output_path}")
        plt.close()


def create_side_by_side_enhancement_grid():
    """Create a side-by-side grid of all enhancements."""
    
    img_dir = Path('img')
    enhanced_dir = Path('enhanced_images')
    
    # Get all image files
    image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    image_files = sorted(image_files)
    
    if not image_files:
        print("No images found")
        return
    
    # Create grid visualization
    fig = plt.figure(figsize=(20, 8 * len(image_files)))
    
    for idx, img_file in enumerate(image_files):
        image_name = img_file.stem
        
        # Load all versions
        original = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        sobel_path = enhanced_dir / 'sobel_enhanced' / f'{image_name}_sobel.png'
        laplacian_path = enhanced_dir / 'laplacian_enhanced' / f'{image_name}_laplacian.png'
        
        if not sobel_path.exists() or not laplacian_path.exists():
            continue
        
        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.imread(str(laplacian_path), cv2.IMREAD_GRAYSCALE)
        
        # Calculate differences
        sobel_diff = cv2.absdiff(original, sobel)
        laplacian_diff = cv2.absdiff(original, laplacian)
        
        row = idx
        
        ax = plt.subplot(len(image_files), 5, row * 5 + 1)
        ax.imshow(original, cmap='gray')
        ax.set_title(f'{image_name} - Original', fontweight='bold')
        ax.axis('off')
        
        ax = plt.subplot(len(image_files), 5, row * 5 + 2)
        ax.imshow(sobel, cmap='gray')
        ax.set_title(f'{image_name} - Sobel', fontweight='bold', color='green')
        ax.axis('off')
        
        ax = plt.subplot(len(image_files), 5, row * 5 + 3)
        ax.imshow(laplacian, cmap='gray')
        ax.set_title(f'{image_name} - Laplacian', fontweight='bold', color='red')
        ax.axis('off')
        
        ax = plt.subplot(len(image_files), 5, row * 5 + 4)
        ax.imshow(sobel_diff, cmap='hot')
        ax.set_title(f'{image_name} - Sobel Diff', fontweight='bold')
        ax.axis('off')
        
        ax = plt.subplot(len(image_files), 5, row * 5 + 5)
        ax.imshow(laplacian_diff, cmap='hot')
        ax.set_title(f'{image_name} - Laplacian Diff', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Edge Enhancement Grid - Original vs Sobel vs Laplacian vs Differences',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = enhanced_dir / 'comparisons' / 'complete_enhancement_grid.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"✓ Complete enhancement grid saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    print("\n[1/2] Creating comprehensive comparison visualizations...")
    create_comprehensive_visualization()
    
    print("\n[2/2] Creating enhancement grid...")
    create_side_by_side_enhancement_grid()
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("="*80)
