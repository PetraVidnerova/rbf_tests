import numpy as np
import argparse
import theano
import theano.tensor as T

import keras
from keras.datasets import mnist 
from keras.models import model_from_json
from keras import backend
from keras.utils import np_utils

from rbflayer import RBFLayer
from initializer import InitFromFile

#from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_th import th_model_eval, batch_eval
from cleverhans.attacks_th import fgsm

import matplotlib.pyplot as plt 


def eval(model_name, X_train, Y_train, X_test, Y_test, cnn=False):

    if not hasattr(backend, "theano"):
        raise RuntimeError("Requires keras to be configured"
                           " to use the Theano backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")

    
    # Define input Theano placeholder
    if cnn:
        x_shape = (None, 28, 28, 1)
        x = T.tensor4('x')
    else:
        x_shape = (None, 784)
        x = T.matrix('x')
        
    y_shape = (None,10)
    y = T.matrix('y')

    
    # load saved model
    print("Load model ... ", end="")
    model = model_from_json(open("models/{}.json".format(model_name)).read(),
                            {'RBFLayer':RBFLayer, 'InitFromFile':InitFromFile})
    model.build(x_shape)
    model.load_weights("models/{}_weights.h5".format(model_name))
    predictions = model(x)
    print("ok")
            
    accuracy = th_model_eval(x, y, predictions, X_test, Y_test, { "batch_size" : 128 })
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval([x], [adv_x], [X_test], { "batch_size" : 128 })
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape
    
    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = th_model_eval(x, y, predictions, X_test_adv, Y_test, { "batch_size" : 128 })
    print('Test accuracy on adversarial examples: ' + str(accuracy)) 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', metavar='model_name', type=str,
                        help='model saved in model_name.json and model_name_weights.h5')
    parser.add_argument('--cnn', action='store_true', help='cnn type network (2d input)')
    
    args = parser.parse_args()
    model_name= args.model_name
    cnn = args.cnn
    
    # Get MNIST test data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Loaded MNIST test data.")

    if cnn:
        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
    else:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    eval(model_name, X_train, Y_train, X_test, Y_test, cnn)
