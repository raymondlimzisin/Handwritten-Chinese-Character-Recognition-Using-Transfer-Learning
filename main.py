import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def getPrediction(filename):
    
    classes = ['光','厂','厉','及','呜','呼','堂','滨','滩','烟','烦','煤','熄','熟','男','突','窗','类','粉','粮']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    #Load model
    model=load_model(r"C:\Users\zisin\Desktop\WebApp\Model\Model G.h5")

    # read in image
    img_path = 'static/images/' + filename
    img = cv2.imread(img_path)
    # resize image to 128x128
    img = np.asarray(Image.open(img_path).resize((128,128)))
    img_list = []
    img_list.append(np.array(img))
    # mean normalization
    img = (img / 255.0)
    x = np.asarray(img_list)
    img = np.expand_dims(img, axis=0)
    # predict on image
    pred = model.predict(x).round()
    # predicted label
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    return pred_class


#test_prediction =getPrediction('example.jpg')
