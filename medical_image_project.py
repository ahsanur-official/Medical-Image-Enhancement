"""
Medical Image Enhancement for Diagnosis Support
================================================
This project implements various image enhancement techniques to improve
the visibility of anatomical structures in medical images (X-rays, MRI, Ultrasound).

Features:
- Noise Reduction (Gaussian, Median, Bilateral Filtering)
- Contrast Enhancement (Histogram Equalization, CLAHE)
- Edge Preservation and Enhancement
- Smoothing and Edge-Preserving Filters
- Sharpening Techniques
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class MedicalImageEnhancer:
    """
    A comprehensive class for medical image enhancement
    """
    
    def __init__(self, image_path):
        """
        Initialize with medical image
        
        Args:
            image_path: Path to the medical image
        """
        self.original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        
    def add_gaussian_noise(self, mean=0, std=10):
        """
        Add Gaussian noise to image (for demonstration purposes)
        
        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of noise
        """
        noise = np.random.normal(mean, std, self.original.shape).astype(np.float32)
        noisy = np.clip(self.original.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    # ===== NOISE REDUCTION TECHNIQUES =====
    
    def gaussian_filter(self, kernel_size=(5, 5), sigma=1.0):
        """
        Apply Gaussian blur for noise reduction
        
        Args:
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
        """
        return cv2.GaussianBlur(self.original, kernel_size, sigma)
    
    def median_filter(self, kernel_size=5):
        """
        Apply median filter - excellent for salt-and-pepper noise
        
        Args:
            kernel_size: Size of median filter kernel
        """
        return cv2.medianBlur(self.original, kernel_size)
    
    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter - reduces noise while preserving edges
        
        Args:
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
        """
        return cv2.bilateralFilter(self.original, d, sigma_color, sigma_space)
    
    def non_local_means_denoising(self, h=10, template_size=7, search_size=21):
        """
        Apply Non-Local Means Denoising - excellent for medical images
        
        Args:
            h: Filter strength
            template_size: Size of template patch
            search_size: Size of search area
        """
        return cv2.fastNlMeansDenoising(self.original, None, h, template_size, search_size)
    
    # ===== CONTRAST ENHANCEMENT TECHNIQUES =====
    
    def histogram_equalization(self):
        """
        Apply standard histogram equalization for contrast enhancement
        """
        return cv2.equalizeHist(self.original)
    
    def clahe(self, clip_limit=2.0, tile_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Better than standard histogram equalization for medical images
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(self.original)
    
    def adaptive_gamma_correction(self, gamma=1.5):
        """
        Apply gamma correction for brightness adjustment
        
        Args:
            gamma: Gamma value (>1 brightens, <1 darkens)
        """
        # Normalize to [0, 1]
        normalized = self.original / 255.0
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        # Convert back to [0, 255]
        return (corrected * 255).astype(np.uint8)
    
    def contrast_stretching(self):
        """
        Apply contrast stretching to utilize full dynamic range
        """
        # Find min and max values
        min_val = np.min(self.original)
        max_val = np.max(self.original)
        
        # Stretch contrast
        if max_val > min_val:
            stretched = ((self.original - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        else:
            stretched = self.original
        
        return stretched
    
    # ===== EDGE ENHANCEMENT TECHNIQUES =====
    
    def unsharp_masking(self, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
        """
        Apply unsharp masking for edge enhancement
        
        Args:
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation
            amount: Strength of sharpening
            threshold: Threshold for sharpening
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(self.original, kernel_size, sigma)
        
        # Calculate sharpened image
        sharpened = cv2.addWeighted(self.original, 1.0 + amount, blurred, -amount, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def laplacian_enhancement(self):
        """
        Apply Laplacian edge enhancement
        """
        # Apply Laplacian
        laplacian = cv2.Laplacian(self.original, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Add to original image
        enhanced = cv2.add(self.original, laplacian)
        
        return enhanced
    
    def sobel_edge_detection(self):
        """
        Apply Sobel edge detection
        """
        # Calculate gradients
        sobel_x = cv2.Sobel(self.original, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.original, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        
        return sobel
    
    def canny_edge_detection(self, threshold1=50, threshold2=150):
        """
        Apply Canny edge detection
        
        Args:
            threshold1: First threshold for hysteresis
            threshold2: Second threshold for hysteresis
        """
        return cv2.Canny(self.original, threshold1, threshold2)
    
    # ===== COMBINED ENHANCEMENT PIPELINE =====
    
    def comprehensive_enhancement(self):
        """
        Apply a comprehensive enhancement pipeline combining multiple techniques
        """
        # Step 1: Noise reduction using bilateral filter
        denoised = self.bilateral_filter(d=9, sigma_color=75, sigma_space=75)
        
        # Step 2: Apply CLAHE for contrast enhancement
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe_obj.apply(denoised)
        
        # Step 3: Edge enhancement using unsharp masking
        enhanced = cv2.GaussianBlur(contrast_enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(contrast_enhanced, 1.5, enhanced, -0.5, 0)
        
        return enhanced
    
    def advanced_enhancement(self):
        """
        Advanced enhancement pipeline optimized for medical images
        """
        # Step 1: Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(self.original, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Step 2: Contrast stretching
        min_val = np.min(denoised)
        max_val = np.max(denoised)
        stretched = ((denoised - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        
        # Step 3: CLAHE with optimized parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(stretched)
        
        # Step 4: Mild sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
        
        return enhanced
    
    # ===== VISUALIZATION AND ANALYSIS =====
    
    def plot_histogram(self, image, title="Histogram"):
        """
        Calculate histogram for visualization
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist
    
    def compare_enhancements(self, save_path=None):
        """
        Compare multiple enhancement techniques side by side
        """
        # Apply various enhancement techniques
        gaussian = self.gaussian_filter()
        median = self.median_filter()
        bilateral = self.bilateral_filter()
        hist_eq = self.histogram_equalization()
        clahe = self.clahe()
        comprehensive = self.comprehensive_enhancement()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        images = [
            (self.original, "Original Image"),
            (gaussian, "Gaussian Filter"),
            (median, "Median Filter"),
            (bilateral, "Bilateral Filter"),
            (hist_eq, "Histogram Equalization"),
            (clahe, "CLAHE"),
            (self.unsharp_masking(), "Unsharp Masking"),
            (comprehensive, "Comprehensive Enhancement")
        ]
        
        for idx, (img, title) in enumerate(images):
            ax = fig.add_subplot(gs[idx // 4, idx % 4])
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle(f'Medical Image Enhancement Comparison - {self.image_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.show()
    
    def analyze_noise_reduction(self, save_path=None):
        """
        Analyze and compare different noise reduction techniques
        """
        # Add noise for demonstration
        noisy = self.add_gaussian_noise(mean=0, std=15)
        
        # Apply different filters
        gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.0)
        median = cv2.medianBlur(noisy, 5)
        bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
        nlm = cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        images = [
            (self.original, "Original (Clean)"),
            (noisy, "Noisy Image"),
            (gaussian, "Gaussian Filter"),
            (median, "Median Filter"),
            (bilateral, "Bilateral Filter"),
            (nlm, "Non-Local Means")
        ]
        
        for ax, (img, title) in zip(axes.flat, images):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle('Noise Reduction Techniques Comparison', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Noise reduction analysis saved to {save_path}")
        
        plt.show()
    
    def analyze_contrast_enhancement(self, save_path=None):
        """
        Analyze and compare different contrast enhancement techniques
        """
        # Apply different techniques
        hist_eq = self.histogram_equalization()
        clahe = self.clahe(clip_limit=2.0)
        clahe_strong = self.clahe(clip_limit=4.0)
        gamma_bright = self.adaptive_gamma_correction(gamma=0.7)
        gamma_dark = self.adaptive_gamma_correction(gamma=1.5)
        stretched = self.contrast_stretching()
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        techniques = [
            (self.original, "Original"),
            (hist_eq, "Histogram Equalization"),
            (clahe, "CLAHE (Clip=2.0)"),
            (clahe_strong, "CLAHE (Clip=4.0)"),
            (gamma_bright, "Gamma Correction (γ=0.7)"),
            (gamma_dark, "Gamma Correction (γ=1.5)"),
            (stretched, "Contrast Stretching"),
            (self.advanced_enhancement(), "Advanced Enhancement")
        ]
        
        for idx, (img, title) in enumerate(techniques):
            # Image
            ax_img = fig.add_subplot(gs[idx // 4, idx % 4])
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title(title, fontsize=11, fontweight='bold')
            ax_img.axis('off')
        
        # Add histograms in the remaining spaces
        ax_hist1 = fig.add_subplot(gs[2, 0])
        ax_hist1.plot(self.plot_histogram(self.original), color='blue')
        ax_hist1.set_title('Original Histogram', fontsize=10)
        ax_hist1.set_xlim([0, 256])
        
        ax_hist2 = fig.add_subplot(gs[2, 1])
        ax_hist2.plot(self.plot_histogram(hist_eq), color='green')
        ax_hist2.set_title('Hist. Eq. Histogram', fontsize=10)
        ax_hist2.set_xlim([0, 256])
        
        ax_hist3 = fig.add_subplot(gs[2, 2])
        ax_hist3.plot(self.plot_histogram(clahe), color='red')
        ax_hist3.set_title('CLAHE Histogram', fontsize=10)
        ax_hist3.set_xlim([0, 256])
        
        ax_hist4 = fig.add_subplot(gs[2, 3])
        ax_hist4.plot(self.plot_histogram(self.advanced_enhancement()), color='purple')
        ax_hist4.set_title('Advanced Histogram', fontsize=10)
        ax_hist4.set_xlim([0, 256])
        
        fig.suptitle(f'Contrast Enhancement Techniques - {self.image_name}', 
                     fontsize=16, fontweight='bold', y=0.99)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Contrast enhancement analysis saved to {save_path}")
        
        plt.show()
    
    def analyze_edge_preservation(self, save_path=None):
        """
        Analyze edge detection and preservation techniques
        """
        # Edge detection methods
        sobel = self.sobel_edge_detection()
        canny = self.canny_edge_detection(50, 150)
        laplacian = cv2.Laplacian(self.original, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Edge-preserving smoothing
        bilateral = self.bilateral_filter()
        bilateral_edges = cv2.Canny(bilateral, 50, 150)
        
        # Enhanced image with edges
        enhanced = self.comprehensive_enhancement()
        enhanced_edges = cv2.Canny(enhanced, 50, 150)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        images = [
            (self.original, "Original Image"),
            (sobel, "Sobel Edge Detection"),
            (canny, "Canny Edge Detection"),
            (laplacian, "Laplacian Edge Detection"),
            (bilateral, "Bilateral Filter (Edge-Preserving)"),
            (bilateral_edges, "Edges After Bilateral"),
            (enhanced, "Comprehensive Enhancement"),
            (enhanced_edges, "Edges After Enhancement")
        ]
        
        for ax, (img, title) in zip(axes.flat, images):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle('Edge Detection and Preservation Analysis', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Edge preservation analysis saved to {save_path}")
        
        plt.show()
    
    def display_complete_pipeline(self, save_path=None):
        """
        Display the complete enhancement pipeline step by step
        """
        # Pipeline steps
        step1 = self.bilateral_filter()  # Noise reduction
        step2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(step1)  # Contrast
        step3 = cv2.GaussianBlur(step2, (0, 0), 3)
        step4 = cv2.addWeighted(step2, 1.5, step3, -0.5, 0)  # Sharpening
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Step-by-step images
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(self.original, cmap='gray')
        ax1.set_title('Step 0: Original Medical Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(step1, cmap='gray')
        ax2.set_title('Step 1: Bilateral Filter (Noise Reduction)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.imshow(step2, cmap='gray')
        ax3.set_title('Step 2: CLAHE (Contrast Enhancement)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.imshow(step4, cmap='gray')
        ax4.set_title('Step 3: Unsharp Masking (Edge Enhancement)', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # Side-by-side comparison
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.imshow(self.original, cmap='gray')
        ax5.set_title('BEFORE: Original Image', fontsize=14, fontweight='bold', color='red')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[2, 2:4])
        ax6.imshow(step4, cmap='gray')
        ax6.set_title('AFTER: Enhanced Image', fontsize=14, fontweight='bold', color='green')
        ax6.axis('off')
        
        fig.suptitle('Complete Medical Image Enhancement Pipeline', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pipeline visualization saved to {save_path}")
        
        plt.show()
    
    def save_enhanced_image(self, output_path, technique='comprehensive'):
        """
        Save enhanced image to file
        
        Args:
            output_path: Path to save the enhanced image
            technique: Enhancement technique to use
        """
        if technique == 'comprehensive':
            enhanced = self.comprehensive_enhancement()
        elif technique == 'advanced':
            enhanced = self.advanced_enhancement()
        elif technique == 'clahe':
            enhanced = self.clahe()
        elif technique == 'histogram':
            enhanced = self.histogram_equalization()
        else:
            enhanced = self.comprehensive_enhancement()
        
        cv2.imwrite(output_path, enhanced)
        print(f"Enhanced image saved to {output_path}")


def main():
    """
    Main function to demonstrate medical image enhancement
    """
    print("=" * 70)
    print("Medical Image Enhancement for Diagnosis Support")
    print("=" * 70)
    print()
    
    # Define image paths
    image_dir = "img"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}\n")
    
    available_images = {
        "1": ("xray.png", "X-Ray Image"),
        "2": ("ultra.png", "Ultrasound Image"),
        "3": (["xray.png", "ultra.png"], "Both Images")
    }
    
    # Display menu
    print("Select an image to process:")
    print("-" * 70)
    print("1. X-Ray Image (xray.png)")
    print("2. Ultrasound Image (ultra.png)")
    print("3. Process Both Images")
    print("-" * 70)
    
    # Get user choice
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in available_images:
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")
    
    # Determine which images to process
    if choice == "3":
        images = ["xray.png", "ultra.png"]
        print(f"\n✓ Selected: Both Images")
    else:
        images = [available_images[choice][0]]
        print(f"\n✓ Selected: {available_images[choice][1]}")
    
    print()
    
    # Process each image
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found. Skipping...")
            continue
        
        print(f"\nProcessing: {img_name}")
        print("-" * 70)
        
        try:
            # Initialize enhancer
            enhancer = MedicalImageEnhancer(img_path)
            
            # Display analysis menu
            print("\nSelect analysis to perform:")
            print("-" * 70)
            print("1. Complete Enhancement Pipeline (Step-by-step)")
            print("2. Compare Multiple Enhancement Techniques")
            print("3. Noise Reduction Analysis")
            print("4. Contrast Enhancement Analysis")
            print("5. Edge Preservation Analysis")
            print("6. Run All Analyses (Complete Demo)")
            print("-" * 70)
            
            # Get analysis choice
            while True:
                analysis_choice = input("\nEnter your choice (1-6): ").strip()
                if analysis_choice in ["1", "2", "3", "4", "5", "6"]:
                    break
                else:
                    print("Invalid choice! Please enter 1-6.")
            
            print()
            
            # Run selected analysis
            if analysis_choice == "1":
                print("Displaying complete enhancement pipeline...")
                enhancer.display_complete_pipeline()
            
            elif analysis_choice == "2":
                print("Comparing multiple enhancement techniques...")
                enhancer.compare_enhancements()
            
            elif analysis_choice == "3":
                print("Analyzing noise reduction techniques...")
                enhancer.analyze_noise_reduction()
            
            elif analysis_choice == "4":
                print("Analyzing contrast enhancement techniques...")
                enhancer.analyze_contrast_enhancement()
            
            elif analysis_choice == "5":
                print("Analyzing edge preservation techniques...")
                enhancer.analyze_edge_preservation()
            
            elif analysis_choice == "6":
                print("Running all analyses...")
                print("\n1. Displaying complete enhancement pipeline...")
                enhancer.display_complete_pipeline()
                
                print("\n2. Comparing multiple enhancement techniques...")
                enhancer.compare_enhancements()
                
                print("\n3. Analyzing noise reduction techniques...")
                enhancer.analyze_noise_reduction()
                
                print("\n4. Analyzing contrast enhancement techniques...")
                enhancer.analyze_contrast_enhancement()
                
                print("\n5. Analyzing edge preservation techniques...")
                enhancer.analyze_edge_preservation()
            
            # Save enhanced image
            print("\nSelect enhancement technique to save:")
            print("-" * 70)
            print("1. Comprehensive Enhancement (Recommended)")
            print("2. Advanced Enhancement")
            print("3. CLAHE Only")
            print("4. Histogram Equalization")
            print("-" * 70)
            
            while True:
                save_choice = input("\nEnter your choice (1-4): ").strip()
                if save_choice in ["1", "2", "3", "4"]:
                    break
                else:
                    print("Invalid choice! Please enter 1-4.")
            
            technique_map = {
                "1": "comprehensive",
                "2": "advanced",
                "3": "clahe",
                "4": "histogram"
            }
            
            output_name = f"enhanced_{img_name}"
            output_path = os.path.join(output_dir, output_name)
            enhancer.save_enhanced_image(output_path, technique=technique_map[save_choice])
            
            print(f"\n✓ Successfully processed {img_name}")
            print(f"✓ Enhanced image saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    print("\n" + "=" * 70)
    print("All images processed successfully!")
    print("=" * 70)
    print("\nKey Learning Outcomes Achieved:")
    print("✓ Enhanced low-contrast medical images")
    print("✓ Applied histogram equalization and CLAHE")
    print("✓ Used smoothing and edge-preserving filters")
    print("✓ Demonstrated the role of image quality in diagnosis")
    print("\nClose all matplotlib windows to exit the program.")


if __name__ == "__main__":
    main()
