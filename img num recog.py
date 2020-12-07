import cv2
import numpy as np
import pickle
imgOriginal = cv2.imread("myData/myData/4/img005-00001.png")

pickle_in=open('model_trained.p','rb')
model=pickle.load(pickle_in)
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

img = np.asarray(imgOriginal)
imgOriginal=cv2.resize(imgOriginal,(560,420))
img = cv2.resize(img,(32,32))
img = preProcessing(img)
img = img.reshape(1,32,32,1)
classIndex = int(model.predict_classes(img))
predictions=model.predict(img)
prob=np.amax(predictions)
if prob>0.8:
    cv2.putText(imgOriginal,str(classIndex)+"|prob:"+str(prob),(0,80),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),4)
cv2.imshow("detected",imgOriginal)
cv2.waitKey(0)