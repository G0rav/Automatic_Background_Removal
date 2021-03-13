import io as IO
import os
import cv2
import numpy as np
from skimage import io
import tensorflow as tf 
import matplotlib.pyplot as plt
from werkzeug.exceptions import BadRequest
from flask import Flask, render_template, request, Response

upload_folder = "static"
app = Flask(__name__)

model = tf.keras.models.load_model('modelcombined_04_0.238711.h5')


@app.route("/", methods= ['POST', 'GET'])
def upload_file():
    
    if request.method == "POST":
        image_file = request.files["image"]
        URL = request.form["url"]
        if image_file:
            image_location = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_location)

            new_image = load_img(image_location)
            new_image_name = 'new_'+image_file.filename+'.jpg'
            new_image_location = save_img(new_image, new_image_name)

            pred = prediction(imgpath=new_image_location, img=None)
            pred_image_name = 'pred_'+image_file.filename
            pred_location = save_img(pred, pred_image_name)            

            return render_template("index.html", message='Your Portrait is Ready.',
                                    image_name = new_image_name, pred_image_name= pred_image_name)
         
        elif URL: 
            downloaded_img = load_img(URL)
            downloaded_img_name = 'download.jpg'
            downloaded_img_location = save_img(downloaded_img, downloaded_img_name)

            pred = prediction(imgpath=downloaded_img_location, img=None)
            pred_image_name = 'pred_'+ downloaded_img_name
            pred_location = save_img(pred, pred_image_name)
            return render_template("index.html", message= 'Your Portrait is Ready.',
                                    image_name = downloaded_img_name, pred_image_name= pred_image_name)

        else:
            return render_template("index.html", message='Please upload an image or Enter image url.',
                                image_name = None, pred_image_name= None)

    else:
        return render_template("index.html", message='Upload an image or Enter image url.',
                                image_name = None, pred_image_name= None)
    


@app.route('/live-stream', methods=["GET","POST"])
def index():
    return render_template('live_stream.html', toggle=None)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:,:,0:3], (256,256), interpolation=cv2.INTER_AREA)
    return img

def save_img(img,image_name):
    """save the image and return the location"""
    image_location = os.path.join(upload_folder, image_name)
    plt.imsave(image_location, img)
    return image_location


def prediction(imgpath=None, img=None):
    if imgpath:
        im = io.imread(imgpath)
    else:
        im = img.copy()
    
    im = cv2.resize(im[:,:,0:3],(256,256))
    
    img = np.array(im)/255
    img = img.reshape((1,)+img.shape)
    pred = model.predict(img)

    p = pred.copy()
    p = p.reshape(p.shape[1:-1])

    p[np.where(p>.3)] = 1
    p[np.where(p<.3)] = 0

    im[:,:,0] = im[:,:,0]*p 
    im[:,:,0][np.where(p!=1)] = 255
    im[:,:,1] = im[:,:,1]*p 
    im[:,:,1][np.where(p!=1)] = 255
    im[:,:,2] = im[:,:,2]*p
    im[:,:,2][np.where(p!=1)] = 255

    return im

vc = cv2.VideoCapture(0)

def gen():
    """Video streaming generator function."""
    while True:
        read_return_code, frame = vc.read()
        frame = cv2.resize(frame, (360, 360), interpolation=cv2.INTER_AREA)

        frame = prediction(imgpath=None, img=frame)

        frame = cv2.resize(frame, (360, 360), interpolation=cv2.INTER_AREA)
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = IO.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400


if __name__ == "__main__":
    app.run()
