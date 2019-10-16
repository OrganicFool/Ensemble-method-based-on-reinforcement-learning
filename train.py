from keras.datasets import cifar10
from policy import *
from model import *
from keras.utils import to_categorical

if __name__ == '__main__':
    """
        we user cifar-10 data set in keras to test the performance of our model
        details of the data set could be found on http://www.cs.toronto.edu/~kriz/cifar.html
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cnn,dense,vgg16=get_model()
    model = Policy((32,32,3),10,cnn,dense,vgg16)
    x_train=x_train.astype('float64')
    x_test=x_test.astype('float64')
    x_train/=255.
    x_test/=255.
    model.train_policy(x_train.reshape((-1,32,32,3)),to_categorical(y_train.reshape(-1),num_classes=10))
    model.save()
    print(model.validate_with_policy(x_test.reshape((-1,32,32,3))).to_categorical(y_test.reshape(-1),num_classes=10))