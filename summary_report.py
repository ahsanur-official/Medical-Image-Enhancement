"""
Summary Report for Edge Enhancement Processing
Shows detailed statistics and verification of all processed images
"""

import cv2
import numpy as np
from pathlib import Path


def analyze_folder(folder_path):
    """Analyze images in a folder."""
    items = list(Path(folder_path).glob('*'))
    return len([i for i in items if i.is_file() and i.suffix in ['.png', '.jpg']])


def get_image_stats(image_path):
    """Get image statistics."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    return {
        'shape': img.shape,
        'mean': np.mean(img),
        'std': np.std(img),
        'contrast': np.std(img) / np.mean(img) if np.mean(img) > 0 else 0
    }


def print_summary():
    """Print summary of all processed images."""
    
    print("\n" + "="*100)
    print("EDGE ENHANCEMENT PROCESSING - COMPLETE SUMMARY")
    print("="*100)
    
    base_dir = Path('enhanced_images')
    
    if not base_dir.exists():
        print("❌ No enhanced_images directory found")
        return
    
    # Check each folder
    sobel_dir = base_dir / 'sobel_enhanced'
    laplacian_dir = base_dir / 'laplacian_enhanced'
    comparison_dir = base_dir / 'comparisons'
    
    sobel_count = analyze_folder(sobel_dir)
    laplacian_count = analyze_folder(laplacian_dir)
    comparison_count = analyze_folder(comparison_dir)
    
    print(f"\n📊 PROCESSING STATISTICS:")
    print(f"   Sobel Enhanced Images:     {sobel_count}")
    print(f"   Laplacian Enhanced Images: {laplacian_count}")
    print(f"   Comparison Visualizations: {comparison_count}")
    
    # List all files
    print(f"\n📁 SOBEL ENHANCED FOLDER:")
    for f in sorted(sobel_dir.glob('*.png')):
        stats = get_image_stats(f)
        if stats:
            print(f"   ✓ {f.name}")
            print(f"      Size: {stats['shape'][0]}x{stats['shape'][1]}")
            print(f"      Contrast: {stats['contrast']:.3f}")
    
    print(f"\n📁 LAPLACIAN ENHANCED FOLDER:")
    for f in sorted(laplacian_dir.glob('*.png')):
        stats = get_image_stats(f)
        if stats:
            print(f"   ✓ {f.name}")
            print(f"      Size: {stats['shape'][0]}x{stats['shape'][1]}")
            print(f"      Contrast: {stats['contrast']:.3f}")
    
    print(f"\n📊 COMPARISONS GENERATED:")
    for f in sorted(comparison_dir.glob('*.png')):
        print(f"   ✓ {f.name}")
    
    print(f"\n✅ ENHANCEMENT COMPLETE!")
    print(f"\n📍 OUTPUT LOCATIONS:")
    print(f"   • Sobel Enhanced:      {sobel_dir}")
    print(f"   • Laplacian Enhanced:  {laplacian_dir}")
    print(f"   • Comparisons:         {comparison_dir}")
    
    print("\n" + "="*100)
    print("KEY ENHANCEMENTS APPLIED:")
    print("="*100)
    print("""
✓ SOBEL EDGE DETECTION:
  - Detects gradient/edge information in both X and Y directions
  - Enhances edges while preserving original image features
  - Better for showing anatomical boundaries
  - Good for detecting rapid intensity changes

✓ LAPLACIAN EDGE DETECTION:
  - Detects second derivatives (zero-crossings)
  - More sensitive to fine details
  - Highlights internal structure transitions
  - Useful for finding peaks and valleys in intensity

✓ EDGE-ENHANCED VISIBILITY:
  - Original images combined with edge maps
  - 50% weight emphasis on detected edges
  - Maintains original anatomical context
  - Enhanced visibility of anatomical structures

✓ COMPARISON VISUALIZATIONS:
  - Side-by-side original vs enhanced comparisons
  - Histogram analysis for each enhancement
  - Edge detection map comparisons
  - Quality metrics displayed

RECOMMENDED USE CASES:
  • Sobel: Use when focused on anatomical boundaries and edges
  • Laplacian: Use when looking for fine structural details
  • Both: Compare to choose the best enhancement for your analysis
""")
    print("="*100)


if __name__ == "__main__":
    print_summary()
