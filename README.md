# Automatic_Background_Remover

A Web API for automatic background removal using Deep Learning. App is made using Flask and deployed on Heroku.

👉 https://portrait-me.herokuapp.com/

Modules used:
```
Numpy
Flask
Matplotlib
Opencv-python
Tensorflow and Keras
```

Datasets used for training:
- COCO - https://cocodataset.org/#home
- Kaggle - https://www.kaggle.com/furkankati/person-segmentation-dataset

### Here is the Quick look at it.

<img src="https://github.com/G0rav/Automatic_Background_Removal/blob/main/src/Website_Screenshot_pc.png" alt="Website Screenshot">


# Model Building:

The model is trained using modified version of U-NET (https://arxiv.org/abs/1505.04597) Architecture first presented by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015.
I have added Residual skip connections in U-NET Model which makes it more robust. 
