import cv2
import  pickle
import numpy as np
#####################3##########3
width = 720
height = 720
threshold=0.8
#################################

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success , imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    img = img.reshape(1,32,32,1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probval = np.amax(predictions)
    print(classIndex,probval)
    if probval>threshold:
        cv2.putText(imgOriginal,str(classIndex)+" Prob:"+str(probval*100)+"%",(50,50),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),3)
    cv2.imshow("detected",imgOriginal)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

