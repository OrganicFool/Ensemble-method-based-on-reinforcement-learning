from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential,Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

INPUT_SHAPE=[32,32,3]
NUM_CLASSES=10

def get_model():
    """
    generate some model for the prediction problem to ensemble

    in this instance, we only use 3 models (a simple CNN, a full-connected network and a VGG16)
    to show how our ensemble model work. your can try more state-of-the-art model on Kaggle or
    other problem.
    :return: a list of each model
    """
    cnn=Sequential()
    cnn.add(Conv2D(32,input_shape=INPUT_SHAPE,kernel_size=[3,3],activation='relu',padding='same'))
    cnn.add(MaxPool2D())
    cnn.add((Conv2D(64,kernel_size=[3,3],activation='relu',padding='same')))
    cnn.add((MaxPool2D()))
    cnn.add(Flatten())
    cnn.add(Dropout(0.2))
    cnn.add(Dense(100,activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(NUM_CLASSES,activation='softmax'))
    cnn.compile('rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    dense = Sequential()
    dense.add(Flatten(input_shape=INPUT_SHAPE))
    dense.add(Dropout(0.2))
    dense.add(Dense(500, activation='relu'))
    dense.add(Dropout(0.2))
    dense.add(Dense(200, activation='relu'))
    dense.add(Dropout(0.2))
    dense.add(Dense(NUM_CLASSES, activation='softmax'))
    dense.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # inception_v3_base_model=InceptionV3(weights=None,input_shape=(28,28,1))
    # x=inception_v3_base_model.output
    # x=Flatten()(x)
    # x=Dense(100,activation='relu')(x)
    # predictions=Dense(10,activation='softmax')(x)
    # inception_v3_model=Model(inception_v3_base_model.input,predictions)
    # inception_v3_model.compile('rmsprop',loss='categorial_crossentropy',metrics=['accuracy'])

    VGG16_base_model = VGG16(weights=None, input_shape=INPUT_SHAPE)
    x = VGG16_base_model.layers[19].output
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    VGG16_model = Model(VGG16_base_model.input, predictions)
    VGG16_model.compile('rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

    return cnn,dense,VGG16_model