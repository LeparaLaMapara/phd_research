# **PhD Research: Predicting Flood Risk and Flood Depth using Self-Supervised Learning (SSL)**

## **1. Research Questions**
### **Primary Question:**
- How can self-supervised learning techniques be used to predict flood risk and flood depth from multimodal datasets, including reanalysis data and satellite imagery?

### **Sub-questions:**
1. What are the most effective pretext tasks in SSL for learning useful representations from reanalysis data and satellite images?
2. How can we integrate multimodal data (satellite images and reanalysis data) to improve flood prediction accuracy?
3. What are the performance gains of using SSL compared to traditional supervised learning methods in flood prediction?
4. How can the models be adapted to different geographic regions with varying flood patterns?

## **2. Literature Review**
- **SSL Techniques:** Review existing SSL techniques such as Contrastive Learning, Predictive Coding, and Masked Image Modeling.
- **Flood Prediction Models:** Survey traditional methods for flood risk and flood depth prediction, focusing on machine learning and remote sensing data.
- **Multimodal Learning:** Investigate strategies for combining satellite imagery and reanalysis data, including early fusion, late fusion, and hybrid approaches.

## **3. Datasets**
### **Reanalysis Data:**
- **ERA5:** Hourly data on a 31 km grid, including precipitation, temperature, wind speed, soil moisture, etc.
- **MERRA-2:** NASA's analysis focusing on climate and weather patterns.
- **GLDAS:** Global Land Data Assimilation System, focusing on land surface states like soil moisture and snow cover.

### **Satellite Images:**
- **Sentinel-1/2:** High-resolution radar and optical imagery suitable for flood detection and land cover analysis.
- **Landsat 8:** Multi-spectral images with 30m resolution, useful for monitoring land changes and water bodies.
- **MODIS:** Daily global data at moderate resolution, useful for large-scale flood analysis.

### **Supplementary Data:**
- **DEM (Digital Elevation Models):** For assessing flood depth and terrain analysis. Sources include SRTM and ALOS PALSAR DEM.
- **Flood Event Archives:** Historical flood data from sources like Dartmouth Flood Observatory or national meteorological agencies.

## **4. Self-Supervised Learning Approaches**
### **Pretext Tasks:**
- **Temporal Consistency:** Predict future frames of satellite imagery or sequence of reanalysis data.
- **Image Reconstruction:** Train an autoencoder to reconstruct satellite images or fill in missing reanalysis data.
- **Contrastive Learning:** Use techniques like SimCLR or MoCo to distinguish between positive (similar) and negative (dissimilar) pairs.
- **Masked Modeling:** Mask portions of satellite images or reanalysis grids and train the model to predict the missing parts.

### **Main Tasks:**
- **Flood Risk Prediction:** Fine-tune the pre-trained model on labeled data to predict flood occurrence probabilities.
- **Flood Depth Estimation:** Use the learned representations to predict continuous flood depth values.

## **5. Methodology**
### **Data Preprocessing:**
- Standardize reanalysis data variables.
- Apply cloud masking and atmospheric correction to satellite images.
- Resample data to a common spatial and temporal resolution.

### **Model Architecture:**
- **CNNs + RNNs/Transformers:** Use CNNs for spatial features from satellite images and RNNs/Transformers for temporal dependencies in reanalysis data.
- **Attention Mechanisms:** Focus on important areas of the image or data sequences that are more likely to indicate flooding.
- **Multimodal Fusion:** Explore early fusion (combine data sources before feeding into the model) and late fusion (combine predictions from separate models trained on different data).

### **Training Procedure:**
- **Pre-training:** Train the model on the pretext task using large amounts of unlabeled data.
- **Fine-tuning:** Use labeled flood data (if available) or semi-supervised techniques for fine-tuning.
- **Validation:** Implement cross-validation and hyperparameter tuning to optimize the model's performance.

### **Evaluation Metrics:**
- **Flood Risk:** Precision, recall, F1 score, and AUC-ROC for classification.
- **Flood Depth:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R² for regression tasks.

## **6. Experimentation**
- **Baseline Models:** Establish a performance baseline with a supervised learning model (e.g., Random Forest or standard CNN).
- **SSL Model:** Implement the SSL-based model and compare its performance with the baseline.
- **Ablation Studies:** Analyze the contribution of different components (e.g., pretext task, data fusion method) to the overall model performance.
- **Domain Adaptation:** Test the model on different geographic regions to assess its generalizability.

## **7. Expected Contributions**
- **Novel SSL Techniques:** Development or adaptation of SSL techniques for flood prediction.
- **Improved Flood Prediction Models:** Demonstrate how multimodal data and SSL can lead to more accurate and reliable flood predictions.
- **Open-Source Tools:** Release datasets, code, and pre-trained models for community use.

## **8. Timeline**
### **Year 1:**
- Conduct literature review and finalize research questions.
- Gather datasets and begin data preprocessing.
- Experiment with simple SSL techniques and pretext tasks.

### **Year 2:**
- Develop and refine the SSL model architecture.
- Perform extensive experimentation and hyperparameter tuning.
- Start writing the first research paper based on preliminary results.

### **Year 3:**
- Focus on model evaluation, including ablation studies and domain adaptation.
- Submit research papers and present findings at conferences.
- Finalize thesis writing and defense preparation.

## **9. Potential Challenges & Mitigation**
- **Data Quality:** Ensure high-quality data preprocessing, especially for satellite imagery, to reduce noise and errors.
- **Computational Resources:** Plan for access to high-performance computing resources, possibly through cloud services or university facilities.
- **Model Complexity:** Regularly review the model's complexity to avoid overfitting, especially when dealing with limited labeled data.

---
