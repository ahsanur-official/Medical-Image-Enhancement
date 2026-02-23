# Interactive Usage Guide

## 🎮 How to Use the Interactive Menus

When you run the program, you'll see three interactive menus:

### Menu 1: Choose Image
```
==================================================================
Medical Image Enhancement for Diagnosis Support
==================================================================

Select an image to process:
----------------------------------------------------------------------
1. X-Ray Image (xray.png)
2. Ultrasound Image (ultra.png)
3. Process Both Images
----------------------------------------------------------------------

Enter your choice (1-3): _
```

**What to choose:**
- `1` - Process only the X-Ray image
- `2` - Process only the Ultrasound image
- `3` - Process both images sequentially

---

### Menu 2: Choose Analysis
```
Processing: xray.png
----------------------------------------------------------------------

Select analysis to perform:
----------------------------------------------------------------------
1. Complete Enhancement Pipeline (Step-by-step)
2. Compare Multiple Enhancement Techniques
3. Noise Reduction Analysis
4. Contrast Enhancement Analysis
5. Edge Preservation Analysis
6. Run All Analyses (Complete Demo)
----------------------------------------------------------------------

Enter your choice (1-6): _
```

**Analysis Options Explained:**

| Option | What It Shows | Best For |
|--------|---------------|----------|
| **1** | Step-by-step enhancement process | Understanding the pipeline |
| **2** | 8 different techniques side-by-side | Comparing methods |
| **3** | Noise reduction filters comparison | Learning about denoising |
| **4** | Contrast enhancement + histograms | Understanding CLAHE vs others |
| **5** | Edge detection and preservation | Seeing edge techniques |
| **6** | All of the above (5 windows) | Complete demonstration |

**Recommended for beginners:** Start with option `1`, then try `6` for full demo.

---

### Menu 3: Choose Enhancement Technique
```
Select enhancement technique to save:
----------------------------------------------------------------------
1. Comprehensive Enhancement (Recommended)
2. Advanced Enhancement
3. CLAHE Only
4. Histogram Equalization
----------------------------------------------------------------------

Enter your choice (1-4): _
```

**Enhancement Techniques:**

| Option | Technique | Description | Best For |
|--------|-----------|-------------|----------|
| **1** | Comprehensive | Bilateral + CLAHE + Sharpening | General use ⭐ |
| **2** | Advanced | NLM + Stretching + CLAHE + Sharpening | Best quality |
| **3** | CLAHE Only | Just CLAHE enhancement | Quick contrast boost |
| **4** | Histogram Eq. | Standard histogram equalization | Academic comparison |

**Recommended:** Option `1` (Comprehensive) for most cases.

---

## 📝 Example Usage Scenarios

### Scenario 1: Quick X-Ray Enhancement
```
Menu 1: Enter 1 (X-Ray only)
Menu 2: Enter 1 (Complete Pipeline)
Menu 3: Enter 1 (Comprehensive)
```
**Result:** Quick enhancement with pipeline visualization

---

### Scenario 2: Complete Demo on Both Images
```
Menu 1: Enter 3 (Both images)
Menu 2: Enter 6 (All analyses)
Menu 3: Enter 1 (Comprehensive)
```
**Result:** Full demonstration on both images (takes longer)

---

### Scenario 3: Focus on Contrast Enhancement
```
Menu 1: Enter 2 (Ultrasound)
Menu 2: Enter 4 (Contrast Analysis)
Menu 3: Enter 3 (CLAHE)
```
**Result:** Learn about contrast techniques, save CLAHE version

---

### Scenario 4: Edge Detection Study
```
Menu 1: Enter 1 (X-Ray)
Menu 2: Enter 5 (Edge Preservation)
Menu 3: Enter 1 (Comprehensive)
```
**Result:** Focus on edge detection methods

---

## 🎯 Sample Run (Step-by-step)

Here's what a complete run looks like:

```bash
python medical_image_project.py
```

**Output:**
```
==================================================================
Medical Image Enhancement for Diagnosis Support
==================================================================

Select an image to process:
----------------------------------------------------------------------
1. X-Ray Image (xray.png)
2. Ultrasound Image (ultra.png)
3. Process Both Images
----------------------------------------------------------------------

Enter your choice (1-3): 1

✓ Selected: X-Ray Image

Processing: xray.png
----------------------------------------------------------------------

Select analysis to perform:
----------------------------------------------------------------------
1. Complete Enhancement Pipeline (Step-by-step)
2. Compare Multiple Enhancement Techniques
3. Noise Reduction Analysis
4. Contrast Enhancement Analysis
5. Edge Preservation Analysis
6. Run All Analyses (Complete Demo)
----------------------------------------------------------------------

Enter your choice (1-6): 1

Displaying complete enhancement pipeline...
[Matplotlib window opens showing 4-step pipeline]

Select enhancement technique to save:
----------------------------------------------------------------------
1. Comprehensive Enhancement (Recommended)
2. Advanced Enhancement
3. CLAHE Only
4. Histogram Equalization
----------------------------------------------------------------------

Enter your choice (1-4): 1

✓ Successfully processed xray.png
✓ Enhanced image saved to: output\enhanced_xray.png

==================================================================
All images processed successfully!
==================================================================

Key Learning Outcomes Achieved:
✓ Enhanced low-contrast medical images
✓ Applied histogram equalization and CLAHE
✓ Used smoothing and edge-preserving filters
✓ Demonstrated the role of image quality in diagnosis

Close all matplotlib windows to exit the program.
```

---

## 💡 Tips for Best Experience

### For Learning:
1. Start with option `6` (Run All Analyses) on **one** image
2. Close each matplotlib window to proceed to next analysis
3. Compare before/after results carefully

### For Quick Results:
1. Choose single image (`1` or `2`)
2. Select option `1` (Complete Pipeline)
3. Save with option `1` (Comprehensive)

### For Detailed Study:
1. Run multiple times with different analyses
2. Try different enhancement techniques
3. Compare saved outputs

### Navigation Tips:
- Close matplotlib windows to continue the program
- Invalid inputs are rejected with helpful messages
- You can run the program multiple times with different choices

---

## ⚡ Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│         QUICK REFERENCE CARD                    │
├─────────────────────────────────────────────────┤
│ MENU 1 - Image Selection                       │
│   1: X-Ray only                                 │
│   2: Ultrasound only                            │
│   3: Both images ⭐                             │
├─────────────────────────────────────────────────┤
│ MENU 2 - Analysis Type                         │
│   1: Pipeline steps                             │
│   2: Compare 8 techniques                       │
│   3: Noise reduction                            │
│   4: Contrast methods                           │
│   5: Edge detection                             │
│   6: Complete demo (all above) ⭐               │
├─────────────────────────────────────────────────┤
│ MENU 3 - Save Technique                        │
│   1: Comprehensive ⭐ (recommended)             │
│   2: Advanced (best quality)                    │
│   3: CLAHE only                                 │
│   4: Histogram equalization                     │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Processing Multiple Images

If you select option `3` (Both Images) in Menu 1:
- You'll go through Menus 2 and 3 for **each** image
- First complete all menus for X-Ray
- Then complete all menus for Ultrasound
- Both enhanced images will be saved

---

## 🎓 Learning Path Recommendation

### Beginner (First Time):
```
Run 1: Menu selections: 1, 1, 1
Run 2: Menu selections: 1, 6, 1
```

### Intermediate:
```
Run 1: Menu selections: 1, 2, 1
Run 2: Menu selections: 1, 4, 3
Run 3: Menu selections: 2, 5, 1
```

### Advanced:
```
Run 1: Menu selections: 3, 6, 1
Run 2: Menu selections: 3, 6, 2
Compare different enhancement techniques
```

---

## 🆘 Troubleshooting

**Q: Can I go back after selecting an option?**
A: No, but you can run the program again with different choices.

**Q: What if I enter an invalid option?**
A: The program will show an error message and ask again.

**Q: How long does option 6 take?**
A: Approximately 30-60 seconds per image (5 analyses).

**Q: Can I skip saving the image?**
A: No, but you can delete the enhanced image afterward if not needed.

---

**Happy Experimenting! 🎨**
