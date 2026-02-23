# Medical Image Enhancement for Diagnosis Support 🏥

A comprehensive image enhancement pipeline for medical images (X-rays, MRI scans, Ultrasound) designed to improve the visibility of anatomical structures and support medical diagnosis.

## 📋 Project Overview

This project implements various state-of-the-art image processing techniques specifically tailored for medical imaging. It addresses key challenges in medical image analysis such as:
- Low contrast in anatomical structures
- Noise from imaging equipment
- Need for edge preservation while enhancing details
- Balancing noise reduction with detail retention

## ✨ Features

### 1. **Noise Reduction Techniques**
- **Gaussian Filter**: Basic smoothing for noise reduction
- **Median Filter**: Excellent for salt-and-pepper noise
- **Bilateral Filter**: Edge-preserving noise reduction
- **Non-Local Means Denoising**: Advanced technique for medical images

### 2. **Contrast Enhancement**
- **Histogram Equalization**: Standard contrast enhancement
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Superior for medical images
- **Gamma Correction**: Brightness adjustment
- **Contrast Stretching**: Dynamic range optimization

### 3. **Edge Preservation & Detection**
- **Unsharp Masking**: Edge enhancement technique
- **Bilateral Filtering**: Smoothing while preserving edges
- **Sobel Edge Detection**: Gradient-based edge detection
- **Canny Edge Detection**: Multi-stage edge detection
- **Laplacian Edge Enhancement**: Second derivative-based detection

### 4. **Comprehensive Pipelines**
- **Comprehensive Enhancement**: Multi-step enhancement pipeline
- **Advanced Enhancement**: Optimized pipeline for medical images
- **Step-by-step Visualization**: Understanding each enhancement stage

## 🚀 Installation

### Requirements
```bash
pip install opencv-python numpy matplotlib
```

### Required Libraries
- **OpenCV (cv2)**: Image processing operations
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization and plotting

## 📂 Project Structure

```
Medical-Image-Enhancement/
│
├── medical_image_project.py    # Main Python script
├── README.md                    # Project documentation
├── QUICKSTART.md                # Quick start guide
├── requirements.txt             # Dependencies
│
├── img/                         # Input images directory
│   ├── xray.png                # X-ray sample image
│   └── ultra.png               # Ultrasound sample image
│
└── output/                      # Enhanced images directory (auto-created)
    ├── enhanced_xray.png       # Enhanced X-ray output
    └── enhanced_ultra.png      # Enhanced ultrasound output
```

## 💻 Usage

### Interactive Mode (Default)

The program features an interactive menu system:

```bash
python medical_image_project.py
```

You'll be prompted with three menus:
1. **Image Selection**: Choose X-Ray, Ultrasound, or both
2. **Analysis Type**: Select specific analysis or run all
3. **Enhancement Technique**: Choose how to save the enhanced image

**Example interaction:**
```
Enter your choice (1-3): 1        # Select X-Ray
Enter your choice (1-6): 6        # Run all analyses
Enter your choice (1-4): 1        # Save comprehensive enhancement
```

See [INTERACTIVE_GUIDE.md](INTERACTIVE_GUIDE.md) for detailed menu explanations.

### Programmatic Usage (Advanced)

```python
from medical_image_project import MedicalImageEnhancer

# Initialize with image path
enhancer = MedicalImageEnhancer("img/xray.png")

# Apply specific techniques
clahe_image = enhancer.clahe(clip_limit=2.0)
bilateral_image = enhancer.bilateral_filter()
enhanced = enhancer.comprehensive_enhancement()

# Visualize comparisons
enhancer.compare_enhancements()
enhancer.analyze_noise_reduction()
enhancer.analyze_contrast_enhancement()
enhancer.analyze_edge_preservation()

# Display complete pipeline
enhancer.display_complete_pipeline()

# Save enhanced image
enhancer.save_enhanced_image("output.png", technique='comprehensive')
```

## 🔬 Enhancement Pipeline

The comprehensive enhancement pipeline follows these steps:

1. **Noise Reduction** (Bilateral Filter)
   - Reduces noise while preserving edges
   - Parameters: d=9, σ_color=75, σ_space=75

2. **Contrast Enhancement** (CLAHE)
   - Improves local contrast
   - Parameters: clipLimit=2.0, tileGridSize=(8,8)

3. **Edge Enhancement** (Unsharp Masking)
   - Sharpens edges and fine details
   - Weighted combination of original and blurred image

## 📊 Analysis Functions

### 1. Complete Pipeline Visualization
```python
enhancer.display_complete_pipeline()
```
Shows step-by-step transformation of the image through the enhancement pipeline.

### 2. Enhancement Techniques Comparison
```python
enhancer.compare_enhancements()
```
Compares 8 different enhancement techniques side-by-side.

### 3. Noise Reduction Analysis
```python
enhancer.analyze_noise_reduction()
```
Demonstrates effectiveness of different noise reduction methods on noisy images.

### 4. Contrast Enhancement Analysis
```python
enhancer.analyze_contrast_enhancement()
```
Compares various contrast enhancement techniques with their histograms.

### 5. Edge Preservation Analysis
```python
enhancer.analyze_edge_preservation()
```
Shows edge detection results before and after enhancement.

## 🎓 Learning Outcomes

By working with this project, you will understand:

1. ✅ **Medical Image Enhancement**
   - How to enhance low-contrast medical images
   - Importance of preserving anatomical details

2. ✅ **Histogram Processing**
   - Standard histogram equalization
   - CLAHE and its advantages over standard methods
   - Histogram analysis for image quality assessment

3. ✅ **Filtering Techniques**
   - Difference between smoothing filters (Gaussian, Median)
   - Edge-preserving filters (Bilateral, NLM)
   - When to use each filter type

4. ✅ **Edge Detection**
   - Various edge detection algorithms
   - Importance of edges in medical diagnosis
   - Trade-offs between noise reduction and edge preservation

5. ✅ **Image Quality in Diagnosis**
   - How image quality affects diagnostic accuracy
   - Role of preprocessing in medical imaging
   - Objective quality metrics

## 🔍 Technical Details

### Key Algorithms Explained

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Divides image into small tiles
- Applies histogram equalization to each tile
- Uses contrast limiting to prevent noise amplification
- **Why for medical images?** Enhances local contrast without over-amplifying noise

#### Bilateral Filter
- Edge-preserving smoothing filter
- Considers both spatial proximity and intensity similarity
- **Formula**: Combines domain filter and range filter
- **Why for medical images?** Reduces noise while preserving important anatomical edges

#### Unsharp Masking
- Subtracts blurred version from original
- Enhances high-frequency components (edges)
- **Process**: Enhanced = Original + α × (Original - Blurred)
- **Why for medical images?** Makes fine details more visible

## 📈 Performance Considerations

- **Processing Time**: Varies by technique
  - Fastest: Gaussian Filter, Histogram Equalization
  - Moderate: Median Filter, CLAHE, Bilateral Filter
  - Slowest: Non-Local Means Denoising

- **Quality vs Speed Trade-off**:
  - For real-time: Use Bilateral + CLAHE
  - For best quality: Use NLM + CLAHE + Unsharp Masking

## 🛠️ Customization

### Adjusting Parameters

#### CLAHE Parameters
```python
# More aggressive enhancement
enhancer.clahe(clip_limit=4.0, tile_size=(4, 4))

# Gentler enhancement
enhancer.clahe(clip_limit=1.5, tile_size=(16, 16))
```

#### Bilateral Filter Parameters
```python
# Stronger noise reduction
enhancer.bilateral_filter(d=15, sigma_color=100, sigma_space=100)

# Better edge preservation
enhancer.bilateral_filter(d=9, sigma_color=50, sigma_space=50)
```

## 📝 Sample Output

The program generates:
1. **Enhanced images** saved in the `output/` directory (auto-created)
2. **Comparison plots** showing different techniques
3. **Analysis visualizations** for each enhancement method
4. **Before/After comparisons** with histograms

## ⚠️ Important Notes

- Medical images should be in grayscale format
- Supported formats: PNG, JPG, JPEG, TIFF
- Enhanced images are for educational purposes
- Always consult medical professionals for diagnosis

## 🔮 Future Enhancements

Potential improvements:
- [ ] Deep learning-based enhancement
- [ ] Real-time video enhancement
- [ ] 3D volume enhancement (MRI, CT scans)
- [ ] Automatic parameter optimization
- [ ] GUI interface for interactive enhancement
- [ ] Quality metrics (PSNR, SSIM) calculation

## 📚 References

1. Pizer, S. M., et al. "Adaptive histogram equalization and its variations" (1987)
2. Tomasi, C., & Manduchi, R. "Bilateral filtering for gray and color images" (1998)
3. Buades, A., et al. "Non-local means denoising" (2005)
4. Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization" (1994)

## 👨‍💻 Author

Created as part of Digital Image Processing course project.

## 📄 License

This project is for educational purposes.

---

**Note**: This software is designed for educational and research purposes only. It should not be used as a replacement for professional medical imaging software or diagnostic tools.
