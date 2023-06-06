from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from PIL import Image #opening, manipulating, and saving many different image file formats
import numpy as np
from skimage import transform #is a collection of algorithms for image processing
import os
import keras

app = Flask(__name__) #This creates a Flask web application instance 
app.config["UPLOAD_FOLDER"] = "static/uploads"

rslt = -1 #These are global variables that will store the result and filename of the uploaded image
file = ""


@app.route('/')
def index():
    return render_template("index.html") # This function returns the rendered HTML template index.html


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global rslt
    if request.method == 'POST':
        f = request.files['myfile']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        pt = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], filename)

        model = keras.models.load_model('model/cancer.h5')

        image = load(pt)
        pred = model.predict(image)
        print(pred)
        c = np.argmax(pred)
        print(c)

        rslt = c
        file = filename
        return render_template("result.html", res=rslt, filep=file)
    else:
        return render_template("index.html")


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
