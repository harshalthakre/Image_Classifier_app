from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# import keras modules 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image #preprocessed on image 
from keras.applications.resnet50 import ResNet50  # pre-trained on images , General purpose classifier

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# defining flask app
app = Flask(__name__)

# Model saved with .h5
MODEL_PATH = 'models/your_model.h5'

model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image_path,model):
	img=image.load_image(image_path,target_size=224,224)

	# preprocessing the image 
	x= image.img_to_array(img)

	x= np.expand_dims(x, axis=0)

	x= preprocess_input(x,mode='caffe')

	preds=model.predict(x)
	return preds

@app.route('/',methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload():
	if request.method== 'POST':
		# get the file from post request
		f = request.files['image']

		# save the file to [FOLDER]
		basepath=os.path.dirname(__file__)
		file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
		f.save(file_path)

		# make predictions 

		preds = model_predict(file_path,model)

		# process your result for human 
		pred_class=decode_predictions(preds,top=1) # decoding the image array 

		result=str(pred_class[0][0][1])

		return result

	return None



if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

