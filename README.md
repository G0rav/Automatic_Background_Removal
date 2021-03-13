# Automatic_Background_Remover

A Web API for automatic background removal using Deep Learning. App is made using Flask and deployed on Heroku.

👉 https://portrait-me.herokuapp.com/


### Here is the Quick look at it.

![](https://github.com/G0rav/Automatic_Background_Removal/blob/main/src/Demo.gif)


# Model Details:

```
CNN Architecture - U-Net with Residual connections
Parameters - 2.2M
Trained on - 153,947 Images
validated on - 2693 Images
batch_size = 32
img_size = (256,256)
Trained for - 4 epochs 
Training time - 80min/epoch on GPUs by Google Colab.

```


Datasets used for training:
- COCO - https://cocodataset.org/#home
- Kaggle - https://www.kaggle.com/furkankati/person-segmentation-dataset


The model is trained using modified version of U-NET (https://arxiv.org/abs/1505.04597) Architecture first presented by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015.
I have added Residual skip connections in U-NET Model which makes it more robust. 

I can't put model architecture here because of its huuge size. [view here.](https://raw.githubusercontent.com/G0rav/Automatic_Background_Removal/main/src/Model%20Architecture.png)

# Result:

```
Training loss - .112
Validation loss - .134

Training accuracy - .941
Validation accuracy - .935
Training meanIOU - .43
Validation meanIOU - .43
```
