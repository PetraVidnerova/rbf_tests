import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.datasets import mnist
from keras.utils import np_utils
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
import argparse


def accuracy_score(y1, y2):
    assert y1.shape == y2.shape
        
    y1_argmax = np.argmax(y1, axis=1)
    y2_argmax = np.argmax(y2, axis=1)
    score = sum(y1_argmax == y2_argmax)
    return (score / len(y1)) * 100
    

def add_rbf_layer(model, betas, X_train, Y_train, X_test, Y_test):
    
    newmodel = Sequential() 
    for i in range(len(model.layers)):
        newmodel.add(model.layers[i])
        
    #    for layer in newmodel.layers:
    #        layer.trainable = False

    rbflayer = RBFLayer(300, betas=betas)
    newmodel.add(rbflayer)
    newmodel.add(Dense(10, use_bias=False, name="dense_rbf"))
    newmodel.add(Activation('softmax', name="Activation_rbf"))

    newmodel.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(),
                     metrics=['acc'])

    newmodel.summary()
    
    #model.compile(loss='mean_squared_error',
    #              optimizer=SGD(lr=0.1, decay=1e-6))

    newmodel.fit(X_train, Y_train,
                 batch_size=128,
                 epochs=3,
                 verbose=1)


    Y_pred = newmodel.predict(X_test)
    print("Test Accuracy: ", accuracy_score(Y_pred, Y_test)) 

    return newmodel 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_model_name', metavar='input', type=str,
                        help='input model saved in input.json and input_weights.h5')
    parser.add_argument('output_model_name', metavar='output', type=str,
                        help='output model saved in output.json and output_weights.h5')
    parser.add_argument('--betas', type=float, help='initial value for betas')
    parser.add_argument('--cnn', action='store_true', help='cnn type network (2d input)')
    
    args = parser.parse_args()
    input_model_name= args.input_model_name
    output_model_name = args.output_model_name
    betas = args.betas if args.betas else 2.0
    cnn = args.cnn

    # load and transform mnist data 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if cnn:
        X_train = X_train.reshape(60000, 28, 28, 1)
    else:     
        X_train = X_train.reshape(60000, 784)
    X_train = X_train.astype('float32')
    X_train /= 255

    if cnn:
        X_test = X_test.reshape(10000, 28, 28, 1)
    else:
        X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # load model from file
    with open("models/{}.json".format(input_model_name)) as f:
        model = model_from_json(f.read()) 
    model.load_weights("models/{}_weights.h5".format(input_model_name)) 

    # create and learn new model 
    newmodel = add_rbf_layer(model, betas, X_train, Y_train, X_test, Y_test)

    # save new model to file 
    name = output_model_name
    json_string = newmodel.to_json()
    open("model/{}.json".format(name),"w").write(json_string) 
    newmodel.save_weights("model/{}_weights.h5".format(name))

    print("---test----")
    m = model_from_json(open("model/{}.json".format(name)).read(), {'RBFLayer':RBFLayer})
    m.load_weights("model/{}_weights.h5".format(name))

    Y_pred = m.predict(X_test)
    print("Test Accuracy: ", accuracy_score(Y_pred, Y_test)) 
