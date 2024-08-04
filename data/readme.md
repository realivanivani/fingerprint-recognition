This is the placeholder directory for the dataset. 

# SOCOFing Dataset

This directory contains the SOCOFing dataset, which is used for fingerprint recognition and biometric system development. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing).

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

         
### Download Instructions

1. **Access the Dataset:**
   - Visit the [Kaggle SOCOFing Dataset Page](https://www.kaggle.com/datasets/ruizgara/socofing?select=SOCOFing) and download the dataset.

2. **Extract the Dataset:**
   - After downloading, extract the dataset into this `dataset/` directory so that it is accessible for processing and model training.

## Data Preparation

Before using the dataset for training models, it may require preprocessing steps such as:

- **Normalization:** Scaling pixel values to a standard range.
- **Augmentation:** Applying transformations like rotations and flips to increase variability.

These preprocessing steps can be automated in the provided Jupyter notebooks in the main repository.

## Licensing and Restrictions

Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/ruizgara/socofing?select=SOCOFing) for specific licensing terms and conditions regarding the use of the dataset. Ensure compliance with any usage restrictions.

