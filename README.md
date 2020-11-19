# Pneumonia detection using CNN
This project aims to detect pneumonia from Chest X-Ray Images.  

# Data:
Kaggle data set is used to train, validate and test the model:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia



# Notes:
The obvious requirement from this model is to have high precision i.e. whevener this model classify the X-ray image as pneumonia, it should be highly confident about the prediction.
-	Various data cleaning steps needed to be done e.g. aspect ratio standardization, image channels adjustment etc.
-	Achieved precision of 94.3 % which was in the acceptable range.


train.py : Script to train the convolutional neural network based model to classify images.                      

predict.ipynb : Jupyter notebook to test the model for a random image
