import os
import cv2
import numpy as np
from skimage import io
import tensorflow as tf 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

upload_folder = "static"
app = Flask(__name__)

model = tf.keras.models.load_model(
    'D:\Gaurav\Projects\Flask\OpenCV\cv2 resizing and showing on web page\models\saved-model-02.h5')

@app.route("/", methods=["GET","POST"])
def upload_file():
    
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_location)

            new_image = load_img(image_location)
            new_image_name = 'new_'+image_file.filename
            new_image_location = os.path.join(upload_folder, new_image_name)
            plt.imsave(new_image_location, new_image)

            pred = prediction(new_image_location)
            pred_image_name = 'pred_'+image_file.filename
            pred_location = os.path.join(upload_folder, pred_image_name)
            plt.imsave(pred_location, pred)
            #cv2.imwrite(pred_location, pred)

            return render_template("index.html", test='Prediction Ready.', image_name = new_image_name, pred_image_name= pred_image_name)
        else:
            return render_template("index.html", test='Please select a file to upload.', 
                                    image_name = None, pred_image_name= None)
   
    return render_template("index.html", test='', image_name = None, pred_image_name= None)

def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
    return img

def prediction(imgpath):
    im = io.imread(imgpath)
    im = cv2.resize(im,(128,128))
    
    img = np.array(im)/255
    img = img.reshape((1,)+img.shape)
    pred = model.predict(img)

    p = pred.copy()
    p = p.reshape(p.shape[1:-1])

    p[np.where(p>.4)] = 1
    p[np.where(p<.4)] = 0

    im[:,:,0] = im[:,:,0]*p 
    im[:,:,0][np.where(p!=1)] = 247
    im[:,:,1] = im[:,:,1]*p 
    im[:,:,1][np.where(p!=1)] = 231
    im[:,:,2] = im[:,:,2]*p
    im[:,:,2][np.where(p!=1)] = 230

    return im

if __name__ == "__main__":
    app.run()
