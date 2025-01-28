# Final Report
---
## Summary of Findings

*Findings are based on the image [basic_cat.jpg](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/basic_cat.jpg)*

### Part 1: Results and Observations
**Top-3 Predictions:**
1. **Tiger Cat:**  &nbsp; &nbsp; &nbsp; &nbsp; ██████████████████ 35% Confidence
2. **Tabby:**  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;█████████████████ 34% Confidence
3. **Egyptian Cat:**  &nbsp;&nbsp;████ 7% Confidence

#### Grad-CAM Visualization
The Grad-CAM heatmap highlights the regions of the image the model focused on to make its predictions.

**Observations:**
- The Grad-CAM heatmap shows that the model primarily focused on the face of the cat, particularly around the nose and mouth area.
- I believe the model effectively identified the main region that would be used to classify the cat in the image.
---
### Part 2: Occlusion Techniques
**Top-3 Predictions for Occlusion Techniques:**
#### 1. Black Box Region Technique
- **Tabby:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;████████████████████ 40% Confidence
- **Tiger Cat:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;██████████████ 28% Confidence
- **Egyptian Cat:** ███████ 13% Confidence

#### 2. Blurred Region Technique
- **Tiger Cat:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;█████████████████ 34% Confidence
- **Tabby:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;████████████████ 31% Confidence
- **Egyptian Cat:** █████████ 18% Confidence

#### 3. Noisy Region Technique
- **Tiger Cat:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;██████████████████████████ 51% Confidence
- **Tabby:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;█████████████████ 34% Confidence
- **Egyptian Cat:** █ 2% Confidence

**Observations:**
 - The black box, blurred region, and noisy region occlusions shifted the model's focus toward the face, particularly the eyes and lower ear area.
 - The noisy region maintained the highest confidence in the top prediction (Tiger Cat), indicating robustness to random noise.
 - **Analysis of Results:**
     - The classifier struggled more with the black box and blurred region occlusions. This could be due to these occlusions blocking key facial features. 
     - The black box occlusion had the greatest impact on performance, causing the largest shift in predictions and reducing confidence levels significantly.

---
## Filter Creation

**3 Filters added to code:**
- Blurred Filter (original code)
- Box Blur Filter
- Contour Filter
- Emboss Filter
- Abstract Filter

**Abstract Filter:**  
This created filter emphasizes bold colors and sharper edges. It creates a vibrant, artistic effect that exaggerates the image's original colors and shapes. By combining techniques such as edge enhancement, increased saturation, contrast, and brightness, the filter transforms the image into a visually "abstract" version.

**Effect on Image:**  
- Enhances fine details, making edges more pronounced.
- Boosts vibrancy by exaggerating the colors for an artistic and dynamic look.
- Useful for creative applications or visualizing how the classifier perceives highlighted features.

---
## Reflection on Working with AI

Working with the AI to explain and write Python code was a highly interactive experience. Throughout the project, there was substantial back-and-forth interaction with the AI to troubleshoot issues in Python code and resolve dependency challenges. This process was crucial in ensuring the program worked effectively and met the project requirements.

Key takeaways include:
- **Explainability:** Grad-CAM heatmaps provided a clear understanding of the classifier's focus regions, revealing how features like a cat's face and ears impact predictions.
- **Sensitivity to Occlusions:** Observing the model's robustness (or lack thereof) to occlusions highlighted areas for improvement in its architecture.
- **Collaborative Problem-Solving:** The AI's guidance and suggestions at times provided more issues than answers. But with persistant attempts and detailed back-and-forth, the AI was able to streamline the troubleshooting process.

---
## Visual Results
Below are the visualizations for the Grad-CAM and occlusion techniques:

| Technique          | Visualization       |
|--------------------|---------------------|
| Original Image     | *[Original Cat Image](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/basic_cat.jpg)*    |
| Grad-CAM           | *[Grad-CAM Comparison](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/grad_cam.png)*  |
| Occlusion Techniques   | *[Occlusion Techniques](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/three_focus_techniques.png)*    |
| Blurred Filter     | *[Blurred Filter](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/blurred_image.png)*    |
| Box Blur Filter       | *[Box Blur Filter](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/box_blur_image.png)*    |
| Contour Filter    | *[Contour Filter](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/contour_image.pnge)*    |
| Emboss Filter    | *[Emboss Filter](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/emboss_image.png)*    |
| Abstract Filter    | *[Abstract Filter](https://github.com/Techdudetony/AI-Image-Processing-Classification/blob/main/abstract_image.png)*    |

---

## Conclusion
The classifier's performance highlights the importance of specific regions in making accurate predictions. The Grad-CAM visualization and occlusion techniques provided valuable insights into the model's interpretability and sensitivity to image alterations. Furthermore, the abstract filter showcases the potential for combining AI and creativity to enhance interpretability and produce visually appealing results.