# Pneumonia-detection
Pneumonia is a lung disease that can be diagnosed by analysing the x-ray film of a person's lungs. This project is an attempt to train a model to detect if a person has pneumonia by analysing the ray images.

# Data:
kaggle data set is considered for training the model:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Notes:
The obvious requirement from this model is to have high precision i.e. whevener this model classify an image as pnematic then it should be highly confident about the prediction.

Chest_x_ray.py : Script to train the convolution neural network based model to classify images.
predict_pneumonia_chest_x_ray.ipynb : Jupyter notebook to test the model for a random image
