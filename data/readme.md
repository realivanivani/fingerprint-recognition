This is the placeholder directory for the dataset. 

# SOCOFing Dataset

This directory contains the SOCOFing dataset, used for fingerprint recognition and biometric system development. The dataset is sourced from Kaggle. All references to the dataset below pertain to the original, unprocessed dataset as made available on Kaggle
[Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing).

However, to use this dataset for machine learning models, we have preprocessed the data using the preprocess.ipynb notebook. This notebook includes steps such as image resizing, label extraction, and saving the data in a format ready for model training. These processed files are stored in NumPy arrays, allowing seamless integration into ML workflows.

Here is the description of the dataset, as writen on their Kaggle's page:

Sokoto Coventry Fingerprint Dataset (SOCOFing) is a biometric fingerprint database designed for academic research purposes. SOCOFing is made up of 6,000 fingerprint images from 600 African subjects and contains unique attributes such as labels for gender, hand and finger name as well as synthetically altered versions with three different levels of alteration for obliteration, central rotation, and z-cut. For a complete formal description and usage policy please refer to the following paper: https://arxiv.org/abs/1807.10609


### Dataset Contents

- **Fingerprint Images:** 
  - Made up of 6,000 fingerprint images from 600 African subjects
  - Contains unique attributes such as labels for gender, hand and finger name (labeled **Real**)
  - It also contains synthetically altered versions with three different levels of alteration for obliteration, central rotation, and z-cut (labeled **Altered**).
  - Images are stored in a structured directory format based on fingerprint classes, Real or Altered.
  - Original images are in .BMP format
  - The dataset is organized as follows:

    1. **/Real:** 
       - Contains 6000 original fingerprint images.
       - Each photo name has labels with
         * Subject's number
         * Gender: M or F
         * Hand: Left or Right
         * Finger: Thumb, Index, Middle, Ring or little.

       - An example: 8__M_Right_index_finger.BMP
         ![8__M_Right_index_finger](https://github.com/user-attachments/assets/4e26c459-b554-43ad-8e17-36a22a8df5bd)

    3. **/Altered:** 
       - Original fingerprint photos with 3 types of alterations: Obliteration, Central rotation and Z-cut
       - Each photo has an additional label of the alteration in it's name: _Obl, _CR, _Z-cut
       - Data are split into 3 directories, based on the level of alterations:
         * **/Altered-Easy**:
           - Containing 17.931 photos 
         * **/Altered-Medium**:
           - Containing 17.067 photos 
         * **/Altered-Hard**:
           - Containing 14.272 photos 
        
        - An example from Altered-Medium dataset: 7__M_Left_thumb_finger_CR.BMP
          ![7__M_Left_thumb_finger_CR](https://github.com/user-attachments/assets/a33b9c1b-66ac-4c09-ad87-2bef03b38efc)

### Preprocessing for Machine Learning

Before using the dataset for machine learning models, it is preprocessed using the preprocess.ipynb notebook. Key steps include:

1.	Resizing: All images are resized to a fixed shape of (96, 96).
2.	Label Extraction: Labels such as subject ID, gender, hand, and finger type are extracted from the filenames.
3.	Storage: Preprocessed images and their corresponding labels are saved into NumPy arrays for easy integration into ML pipelines.

The files are saved as:
	* x_real.npy and y_real.npy for the real images.
	* x_easy.npy, x_medium.npy, and x_hard.npy for the altered images (Easy, Medium, and Hard).

These preprocessing steps ensure the dataset is prepared for model training, reducing the need for on-the-fly processing during model execution.

         
### Download Instructions

1. **Access the Dataset:**
   - Visit the [Kaggle SOCOFing Dataset Page](https://www.kaggle.com/datasets/ruizgara/socofing?select=SOCOFing) and download the zip file.

2. **Extract the Dataset:**
   - After downloading, extract the dataset into this `dataset/` directory so that it is accessible for processing and model training.
   - You will have 2 folders, /Real and /Altered
   - /Altered will have 3 subfolders:
         * /Altered-Easy
         * /Altered-Medium
         * /Altered-Hard

3.	**Run Preprocessing:**
	-	Use the provided preprocess.ipynb notebook to preprocess the images and labels for model training.


## Licensing and Restrictions

Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/ruizgara/socofing?select=SOCOFing) for specific licensing terms and conditions regarding the use of the dataset. Ensure compliance with any usage restrictions.

