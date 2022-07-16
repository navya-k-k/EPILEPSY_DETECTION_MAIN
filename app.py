#Importing all the libraries needed
import tensorflow as tf



import os
import pandas as pd





import numpy as np



model_load =tf.keras.models.load_model(r'model_VGG.h5',compile=False)

# Flask utils
from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer





# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
    
        # Get the file from post request
        print(request.files)
        f = request.files['image']
      
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("Base--",basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img = tf.keras.preprocessing.image.load_img(file_path, grayscale=False, target_size=(128, 128))
        x = tf.keras.preprocessing.image.img_to_array(img)
        print(x)
        x = np.expand_dims(x, axis=0)
        x = np.array(x, 'float32')

        print(x.shape)
        # x /= 255
     
        preds=model_load.predict(x)
        print("prediction",preds)
  
        pred=preds[0][0]
      
        res={}
        if(preds==0):
            res['status']=0
        else:
            res['status']=1
        return res

if __name__ == '__main__':
   app.run(debug=True,port=5001)