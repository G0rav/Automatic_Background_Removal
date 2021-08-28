# Automatic_Background_Remover

A Web API for automatic background removal using Deep Learning. App is made using Flask and deployed on Heroku.

👉 https://portrait-me.herokuapp.com/


### Here is the Quick look at it.

![](https://github.com/G0rav/Automatic_Background_Removal/blob/main/src/Demo.gif)


# Model Details:

```
CNN Architecture - U-Net with Residual connections
Parameters - 8.9M
Trained on - 64,115 Images
validated on - 2693 Images
batch_size = 32
img_size = (256,256)
Trained for - 13 epochs 
Training time - 80min/epoch on GPUs by Google Colab.

```


Datasets used for training:
- COCO - https://cocodataset.org/#home

The model is trained using modified version of U-NET (https://arxiv.org/abs/1505.04597) Architecture first presented by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015.
I have added Residual skip connections in U-NET Model which makes it more robust. 

I can't put model architecture here because of its huuge size. [view here.](https://raw.githubusercontent.com/G0rav/Automatic_Background_Removal/main/src/Model%20Architecture.png)

# Result:

```
Training loss - 0.038
Validation loss - 0.056

Training accuracy - 0.935
Validation accuracy - 0.907
Training meanIOU - 0.817
Validation meanIOU - 0.787
```
<img src="https://github.com/G0rav/Automatic_Background_Removal/blob/main/src/Evaluation_Chart.png">
