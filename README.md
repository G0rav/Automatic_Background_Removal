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

Currently the API works really well on images, but not on images.However, Real time image segmentation API works great on local system. 

### Here is the Quick look at it.

<img src="https://github.com/G0rav/Automatic_Background_Removal/blob/main/src/Website_Screenshot_pc.png" alt="Website Screenshot">


# Model Building:

The model is trained using modified version of U-NET (https://arxiv.org/abs/1505.04597) Architecture first presented by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015.
I have added Residual skip connections in U-NET Model which makes it more robust. 

I can't put model architecture here because of its huuge size. [view here.](https://raw.githubusercontent.com/G0rav/Automatic_Background_Removal/main/src/Model%20Architecture.png)

It is trained only upto 4 epoch as of now on image size of 256x256.
