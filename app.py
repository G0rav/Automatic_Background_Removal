import os
import cv2
import numpy as np
from skimage import io
import tensorflow as tf 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

upload_folder = "static"
app = Flask(__name__)

model = tf.keras.models.load_model('saved-model-02.h5')


@app.route("/", methods= ['POST', 'GET'])
def upload_file():
    
    if request.method == "POST":
        image_file = request.files["image"]
        URL = request.form["url"]
        if image_file:
            image_location = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_location)

            new_image = load_img(image_location)
            new_image_name = 'new_'+image_file.filename
            new_image_location = save_img(new_image, new_image_name)

            pred = prediction(new_image_location)
            pred_image_name = 'pred_'+image_file.filename
            pred_location = save_img(pred, pred_image_name)            

            return render_template("index.html", message='Prediction Ready.',
                                    image_name = new_image_name, pred_image_name= pred_image_name)
         
        elif URL:
            #URL = request.form["url"] 
            print('*'*50,URL)
            downloaded_img = load_img(URL)
            downloaded_img_name = URL[-10:-6]+'.jpg'
            downloaded_img_location = save_img(downloaded_img, downloaded_img_name)

            pred = prediction(downloaded_img_location)
            pred_image_name = 'pred_'+ downloaded_img_name
            pred_location = save_img(pred, pred_image_name)
            return render_template("index.html", message= 'Prediction Ready.',
                                    image_name = downloaded_img_name, pred_image_name= pred_image_name)

        else:
            return render_template("index.html", message='Please upload an image or Enter image url.',
                                image_name = None, pred_image_name= None)

    else:
        return render_template("index.html", message='Please upload an image or Enter image url.',
                                image_name = None, pred_image_name= None)
    

def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:,:,0:3], (128,128), interpolation=cv2.INTER_AREA)
    return img

def save_img(img,image_name):
    """save the image and return the location"""
    image_location = os.path.join(upload_folder, image_name)
    plt.imsave(image_location, img)
    return image_location


def prediction(imgpath):
    im = io.imread(imgpath)
    im = cv2.resize(im[:,:,0:3],(128,128))
    
    img = np.array(im)/255
    img = img.reshape((1,)+img.shape)
    pred = model.predict(img)

    p = pred.copy()
    p = p.reshape(p.shape[1:-1])

    p[np.where(p>.4)] = 1
    p[np.where(p<.4)] = 0

    im[:,:,0] = im[:,:,0]*p 
    im[:,:,0][np.where(p!=1)] = 255
    im[:,:,1] = im[:,:,1]*p 
    im[:,:,1][np.where(p!=1)] = 255
    im[:,:,2] = im[:,:,2]*p
    im[:,:,2][np.where(p!=1)] = 255

    return im

if __name__ == "__main__":
    app.run()
