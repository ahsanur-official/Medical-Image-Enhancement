"""
Medical Image Enhancement - Usage Examples

This file demonstrates various use cases and practical examples
for the medical image enhancement pipeline.
"""

from medical_image_enhancement import MedicalImageEnhancer
import numpy as np


# ============================================================================
# EXAMPLE 1: Basic Enhancement of a Synthetic Image
# ============================================================================

def example_1_basic_enhancement():
    """
    Demonstrates basic image enhancement with a synthetic chest X-ray.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Enhancement of Synthetic Image")
    print("="*70)
    
    # Create enhancer and synthetic image
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type='chest')
    
    # Apply complete pipeline
    enhanced = enhancer.apply_complete_enhancement_pipeline()
    
    # Display statistics
    print("\nOriginal vs Enhanced Statistics:")
    stats = enhancer.compare_enhancements()
    
    print(f"Contrast Ratio - Original: {stats['original']['contrast']:.3f}")
    print(f"Contrast Ratio - Enhanced: {stats['enhanced']['pipeline']['contrast']:.3f}")
    
    # Save visualization
    enhancer.visualize_comparison(enhanced_key='pipeline', 
                                 save_path='example1_comparison.png')
    print("\n✓ Example 1 complete - See example1_comparison.png")


# ============================================================================
# EXAMPLE 2: Comparing Different Denoising Methods
# ============================================================================

def example_2_compare_denoising():
    """
    Compare different noise reduction techniques.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Denoising Methods")
    print("="*70)
    
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type='xray')
    
    methods = ['bilateral', 'morphological', 'nlm']
    
    print("\nDenoising Methods Comparison:")
    print(f"{'Method':<20} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 56)
    
    # Original stats
    stats_orig = enhancer.get_image_statistics()
    print(f"{'Original':<20} {stats_orig['mean']:>12.2f} "
          f"{stats_orig['std']:>12.2f} {stats_orig['contrast']:>12.3f}")
    
    # Test each method
    for method in methods:
        denoised = enhancer.apply_noise_reduction(method=method)
        stats = enhancer.get_image_statistics(denoised)
        print(f"{method.capitalize():<20} {stats['mean']:>12.2f} "
              f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}")
    
    print("\n✓ Example 2 complete")


# ============================================================================
# EXAMPLE 3: CLAHE Parameter Tuning
# ============================================================================

def example_3_clahe_tuning():
    """
    Demonstrate the effect of CLAHE parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: CLAHE Parameter Tuning")
    print("="*70)
    
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type='mri')
    
    print("\nEffect of clipLimit (tile size = 8x8):")
    print(f"{'ClipLimit':<15} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 51)
    
    clip_limits = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for clip in clip_limits:
        enhanced = enhancer.apply_clahe(clip_limit=clip, tile_size=8)
        stats = enhancer.get_image_statistics(enhanced)
        print(f"{clip:<15.1f} {stats['mean']:>12.2f} "
              f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}")
    
    print("\nEffect of tileGridSize (clipLimit = 2.0):")
    print(f"{'TileSize':<15} {'Mean':>12} {'Std Dev':>12} {'Contrast':>12}")
    print("-" * 51)
    
    tile_sizes = [4, 8, 16, 32]
    
    for tile in tile_sizes:
        enhanced = enhancer.apply_clahe(clip_limit=2.0, tile_size=tile)
        stats = enhancer.get_image_statistics(enhanced)
        print(f"{tile}x{tile:<12} {stats['mean']:>12.2f} "
              f"{stats['std']:>12.2f} {stats['contrast']:>12.3f}")
    
    print("\n✓ Example 3 complete")
    print("\nRecommendation:")
    print("  - For subtle structures: Use lower clipLimit (1.5-2.0)")
    print("  - For pronounced structures: Use higher clipLimit (3.0-4.0)")
    print("  - Smaller tiles: More local adaptation (better for varied anatomy)")
    print("  - Larger tiles: More global adaptation (smoother result)")


# ============================================================================
# EXAMPLE 4: Edge Detection Comparison
# ============================================================================

def example_4_edge_detection():
    """
    Compare different edge detection methods.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Edge Detection Methods")
    print("="*70)
    
    enhancer = MedicalImageEnhancer()
    enhancer.create_sample_medical_image(image_type='xray')
    
    # Apply enhancement first for better edge detection
    enhancer.apply_complete_enhancement_pipeline()
    enhanced = enhancer.enhanced_images['pipeline']
    
    methods = ['canny', 'sobel', 'laplacian']
    edge_counts = []
    
    print("\nEdge Detection Statistics:")
    print(f"{'Method':<15} {'Edge Pixels':>15} {'Percentage':>12}")
    print("-" * 45)
    
    total_pixels = enhanced.shape[0] * enhanced.shape[1]
    
    for method in methods:
        edges = enhancer.detect_edges(method=method)
        edge_pixels = np.count_nonzero(edges)
        percentage = (edge_pixels / total_pixels) * 100
        edge_counts.append(edge_pixels)
        
        print(f"{method.capitalize():<15} {edge_pixels:>15} {percentage:>11.2f}%")
    
    print("\n✓ Example 4 complete")
    print("\nMethod Characteristics:")
    print("  - Canny: Most selective, best for clear boundaries")
    print("  - Sobel: Detects gradient magnitude, good for varying edges")
    print("  - Laplacian: Detects zero-crossings, sensitive to noise")


# ============================================================================
# EXAMPLE 5: Complete Pipeline Analysis
# ============================================================================

def example_5_complete_analysis():
    """
    Perform complete analysis on all synthetic image types.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Analysis - All Image Types")
    print("="*70)
    
    image_types = ['xray', 'mri', 'chest']
    
    for img_type in image_types:
        print(f"\n{img_type.upper()} Analysis:")
        print("-" * 50)
        
        enhancer = MedicalImageEnhancer()
        enhancer.create_sample_medical_image(image_type=img_type)
        
        # Apply all techniques
        enhancer.apply_noise_reduction(method='bilateral')
        enhancer.apply_histogram_equalization()
        enhancer.apply_clahe(clip_limit=2.0, tile_size=8)
        enhancer.apply_adaptive_histogram_equalization()
        enhancer.apply_edge_preservation(sigma=1.0)
        enhancer.apply_complete_enhancement_pipeline()
        
        # Get statistics
        stats = enhancer.compare_enhancements()
        
        print(f"\nOriginal Image:")
        print(f"  Mean:        {stats['original']['mean']:>10.2f}")
        print(f"  Std Dev:     {stats['original']['std']:>10.2f}")
        print(f"  Contrast:    {stats['original']['contrast']:>10.3f}")
        print(f"  Min-Max:     {stats['original']['min']:>10.0f} - {stats['original']['max']:<.0f}")
        
        print(f"\nBest Enhancement (Pipeline):")
        pipeline_stats = stats['enhanced']['pipeline']
        print(f"  Mean:        {pipeline_stats['mean']:>10.2f}")
        print(f"  Std Dev:     {pipeline_stats['std']:>10.2f}")
        print(f"  Contrast:    {pipeline_stats['contrast']:>10.3f}")
        print(f"  Min-Max:     {pipeline_stats['min']:>10.0f} - {pipeline_stats['max']:<.0f}")
        
        # Calculate improvements
        contrast_improvement = (
            (pipeline_stats['contrast'] - stats['original']['contrast']) / 
            stats['original']['contrast'] * 100
        ) if stats['original']['contrast'] > 0 else 0
        
        print(f"\nImprovement:")
        print(f"  Contrast:    {contrast_improvement:>10.2f}%")
    
    print("\n✓ Example 5 complete")


# ============================================================================
# EXAMPLE 6: Custom Image Enhancement
# ============================================================================

def example_6_custom_enhancement():
    """
    Demonstrate how to use the enhancer with custom numpy arrays.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Image Enhancement")
    print("="*70)
    
    # Create a custom image (e.g., from camera or other source)
    custom_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # Create enhancer with custom array
    enhancer = MedicalImageEnhancer(image_array=custom_image)
    
    # Apply enhancement
    enhanced = enhancer.apply_complete_enhancement_pipeline()
    
    # Analyze
    print("\nCustom Image Enhancement:")
    stats = enhancer.compare_enhancements()
    
    print(f"Original - Mean: {stats['original']['mean']:.2f}, "
          f"Contrast: {stats['original']['contrast']:.3f}")
    print(f"Enhanced - Mean: {stats['enhanced']['pipeline']['mean']:.2f}, "
          f"Contrast: {stats['enhanced']['pipeline']['contrast']:.3f}")
    
    print("\n✓ Example 6 complete")


# ============================================================================
# EXAMPLE 7: Batch Processing
# ============================================================================

def example_7_batch_processing():
    """
    Demonstrate batch processing of multiple images.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Batch Processing")
    print("="*70)
    
    image_types = ['xray', 'mri', 'chest']
    batch_results = []
    
    print("\nBatch Processing Results:")
    print(f"{'Image Type':<15} {'Original Contrast':>20} "
          f"{'Enhanced Contrast':>20}")
    print("-" * 57)
    
    for img_type in image_types:
        enhancer = MedicalImageEnhancer()
        enhancer.create_sample_medical_image(image_type=img_type)
        enhancer.apply_complete_enhancement_pipeline()
        
        stats = enhancer.compare_enhancements()
        orig_contrast = stats['original']['contrast']
        enh_contrast = stats['enhanced']['pipeline']['contrast']
        
        batch_results.append({
            'type': img_type,
            'original': orig_contrast,
            'enhanced': enh_contrast
        })
        
        print(f"{img_type:<15} {orig_contrast:>20.3f} {enh_contrast:>20.3f}")
    
    print("\n✓ Example 7 complete - Batch processing successful")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("MEDICAL IMAGE ENHANCEMENT - PRACTICAL EXAMPLES")
    print("*" * 70)
    
    # Run examples
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
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
