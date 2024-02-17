import numpy as np
import os
from flask import Flask, app,request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for,flash
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
modeln=load_model(r"plant_type_classifier.h5")
#default home page or route
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index():
    return render_template ('predict.html')


@app.route('/predict',methods=["GET","POST"])
def nres():
    if request.method== "POST":
        if 'image' not in request.files:
            return redirect('index')
        else:
            f=request.files['image']
            basepath=os.path.dirname(__file__) #getting the current path i.e where app-py is present
            #print ("current path",basepath)
            filepath=os.path.join(basepath,UPLOAD_FOLDER ,f.filename) #from anywhere in the system we can give image but we want t
            #print ("upload folder is", filepath)
            f.save(filepath)
            img=image.load_img(filepath,target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            #print(x)
            img_data=preprocess_input(x)
            prediction=np.argmax(modeln.predict (img_data))
            index = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3','Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2','Grape White 1','Grape White 2','Grape White 3','Grape White 4','Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge']
       #      index=['Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata', 'beau_s lines', 
       #      'bluish nail','clubbing', 'eczema', 'half and half nailes (Lindsay_s nails)', 
       #      'koilonychia', 'leukonychia','onycholycis', 'pale nail', 'red lunula', 
       #          'splinter hemmorrage', 'terry_s nail', 'white nail', 'yellow nails']
            result = str(index[prediction])
            return render_template('result.html',result=result)
    return redirect(url_for('index'))

"""Running our application"""
if __name__ == "__main__":
    app.run(debug =True,port = 8080)
# import numpy as np
# import pandas as pd
# import os
# import tensorflow as tf
# from flask import Flask, app, request, render_template
# from keras.models import Model
# from keras.preprocessing import image
# from tensorflow.python.ops.gen_array_ops import Concat
# from keras.models import load_model
# model=load_model('plant_type_classifier.h5') 
# app = Flask (__name__)
# @app.route('/')
# def Grapevine():
#     return render_template('predict.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#        result =''
#        if request.method=="POST":
#               f=request.files['image']
#        basepath=os.path.dirname(__file__)
#        filepath=os.path.join(basepath, 'uploads', f.filename)
#        f.save(filepath)
#        img = tf.keras.utils.load_img(filepath, target_size=(128,128))
#        x = tf.keras.utils.img_to_array(img)
#        x = np.expand_dims(x,axis=0)
#        prediction= np.argmax(model.predict(img), axis=1)
#        op= ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3','Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2','Grape White 1','Grape White 2','Grape White 3','Grape White 4','Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge']
#        result= str(op[prediction[0]])
#        print(result)
#        return render_template('predict.html',result=result)
# if __name__== '__main__':
#     app.run(debug=True)