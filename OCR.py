import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

##############################
path="myData/myData"
pathLabels = 'labels.csv'
test_ratio=0.2
validation_ratio=0.2
imageDimensions = (32,32,3)
batchsizeval=50
epochsval = 10
stepsforepoch = 2000

##############################
images=[]
classNo=[]
noofSamples=[]
myList=os.listdir(path)
print("Total number of classes detected",len(myList))
print('importing Classes .........')
noofClasses=len(myList)
for x in range(noofClasses):
    myPicList=os.listdir(path+'/'+str(x))
    for y in myPicList:
        currentImg=cv2.imread(path+'/'+str(x)+'/'+y)
        currentImg=cv2.resize(currentImg,(imageDimensions[0],imageDimensions[1]))
        images.append(currentImg)
        classNo.append(x)
    print(x,end=' ')
print(' ')
print('total images in images List: ',len(images))
print('total ID\'s in classNo list: ',len(classNo))
images=np.array(images)
classNo=np.array(classNo)
print(images.shape)
x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=test_ratio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=validation_ratio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
for x in range(0,noofClasses):
    noofSamples.append(len(np.where(y_train==x)[0]))
print(noofSamples)
plt.figure(figsize=(10,5))
plt.bar(range(0,noofClasses),noofSamples)
plt.title('No of images for each Class')
plt.xlabel("Class ID")
plt.ylabel('Number of images')
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

x_train=np.array(list(map(preProcessing,x_train)))
x_test=np.array(list(map(preProcessing,x_test)))
x_validation=np.array(list(map(preProcessing,x_validation)))

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train,noofClasses)
y_test = to_categorical(y_test,noofClasses)
y_validation = to_categorical(y_validation,noofClasses)

def myModel():
    nooffilters = 60
    sizeofFilter1 = (5,5)
    sizeofFilter2 = (3,3)
    sizeofPool = (2,2)
    noofNode = 500

    model = Sequential()
    model.add((Conv2D(nooffilters,sizeofFilter1,input_shape=(imageDimensions[0],
                                                             imageDimensions[1],
                                                             1),activation='relu')))
    model.add((Conv2D(nooffilters,sizeofFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add((Conv2D(nooffilters//2,sizeofFilter2,activation='relu')))
    model.add((Conv2D(nooffilters//2,sizeofFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noofNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noofClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model = myModel()
#print(model.summary())
print(x_train.shape)
print(x_validation.shape)
print(y_train.shape)
print(y_validation.shape)


history = model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batchsizeval),
                    epochs=epochsval,
                    validation_data=(x_validation,y_validation),
                    shuffle=1)

score=model.evaluate(x_test,y_test,verbose=0)
print('Test score',score[0])
print('Test Accuracy',score[1])

pickle_out=open("model_trained.p",'wb')
pickle.dump(model,pickle_out)
pickle_out.close()
