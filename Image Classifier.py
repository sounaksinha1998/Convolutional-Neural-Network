import numpy as np
import cv2
import random
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

label = {"dog":np.array([1,0],np.float32),"cat":np.array([0,1],np.float32)}
n_classes = 2

def image_resize(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (32,32))
    return img

def image_augmentation(img,hor,ver,ang,zm):
    col,row,chanel = img.shape
    #translation:
    matrix = np.float32( [[1,0,hor],[0,1,ver]] )
    img = cv2.warpAffine(img, matrix, (row,col))
    #rotation and zoom:
    matrix = cv2.getRotationMatrix2D((row/2,col/2), ang, zm)
    img = cv2.warpAffine(img, matrix, (row,col))
    return img

def create_training_dataset(sample_size):
    ipdata=[]
    oplabels=[]
    for i in range(sample_size):
        dog = np.float32( image_resize( "D:\\Image_Dataset\\train\\dog\\dog."+str(i)+".jpg" ) )
        dogaug = image_augmentation(dog,0,0,random.uniform(0,45),random.uniform(1,2))
        ipdata.append(dog)
        ipdata.append(dogaug)
        oplabels.append(label["dog"])
        oplabels.append(label["dog"])

        cat = np.float32(image_resize("D:\\Image_Dataset\\train\\cat\\cat." + str(i) + ".jpg"))
        cataug = image_augmentation(cat,0,0,random.uniform(0,45),random.uniform(1,2))
        ipdata.append(cat)
        ipdata.append(cataug)
        oplabels.append(label["cat"])
        oplabels.append(label["cat"])

    ipdata = np.asarray(ipdata)
    oplabels = np.asarray(oplabels)

    dataset = np.c_[ipdata.reshape( len(ipdata),-1 ), oplabels.reshape( len(oplabels),-1 )]
    random.shuffle(dataset)

    ipdata = dataset[:,0:32*32*3].reshape(-1,32,32,3) //255
    oplabels = dataset[:,32*32*3:32*32*3+n_classes]

    return ipdata, oplabels

def conv_net(X,y,batch_size,epochs):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,trainable=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes,trainable=True))
    model.add(Activation("softmax"))

    #sgd = keras.optimizers.sgd(lr=0.1, decay=1e-6, momentum=0.5)
    #rms = keras.optimizers.rmsprop(lr=0.1, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model

ipdata,oplabels = create_training_dataset(12000)
model = conv_net(ipdata,oplabels,32,10)
model.save("Cifar10CNN4.h5")
