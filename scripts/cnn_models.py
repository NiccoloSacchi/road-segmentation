""" Here we put all our cnn models """

# import 'Sequential' is a linear stack of neural network layers. Will be used to build the feed-forward CNN
from keras.models import Sequential 
# import the "core" layers from Keras (these are the most common layers)
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
# import the convolutional layers that will help us efficiently train on image data
from keras.layers import Conv2D, MaxPooling2D

def model1():
    # wiht leaky relu
    nclasses = 2
    model = Sequential()

    # layer 1
    model.add(
        Conv2D(32, (11, 11), 
               padding="same", 
               input_shape=(None, None, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25)) 

    # layer 2
    model.add(
        Conv2D(48, (5, 5),
               strides=(2, 2),
               padding="same"
              ))
    model.add(LeakyReLU(alpha=0.1))

    # # later 3
    # model.add(
    #     Conv2D(48, (5, 5), 
    #            padding="same")) 
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.25)) 

    # layer 4
    model.add(
        Conv2D(48, (5, 5),
               strides=(2, 2),
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    # # layer 5
    # model.add(
    #     Conv2D(48, (5, 5), 
    #            padding="same")) 
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.25)) 

    # layer 6
    model.add(
        Conv2D(64, (5, 5), 
               strides=(2, 2),
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    # # layer 7
    # model.add(
    #     Conv2D(64, (5, 5), 
    #            padding="same")) 
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.25)) 

    # layer 8
    model.add(
        Conv2D(2, (5, 5), 
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Activation('softmax'))
    
    print(model.summary())
    return model

def model2():
    # with relu
    nclasses = 2
    model = Sequential()

    # layer 1
    model.add(
        Conv2D(32, (11, 11), 
               activation='relu',
               padding="same", 
               input_shape=(None, None, 3)))
    model.add(Dropout(0.25)) 

    # layer 2
    model.add(
        Conv2D(48, (5, 5),
               activation='relu',
               strides=(2, 2),
               padding="same"
              ))

    # later 3
    model.add(
        Conv2D(48, (5, 5), 
               activation='relu',
               padding="same")) 
    model.add(Dropout(0.25)) 

    # layer 4
    model.add(
        Conv2D(48, (5, 5),
               activation='relu',
               strides=(2, 2),
               padding="same"))

    # layer 5
    model.add(
        Conv2D(48, (5, 5), 
               activation='relu',
               padding="same")) 
    model.add(Dropout(0.25)) 

    # layer 6
    model.add(
        Conv2D(64, (5, 5), 
               activation='relu',
               strides=(2, 2),
               padding="same"))

    # layer 7
    model.add(
        Conv2D(64, (5, 5), 
               activation='relu',
               padding="same")) 
    model.add(Dropout(0.25)) 

    # layer 8
    model.add(
        Conv2D(2, (5, 5), 
               activation='relu',
               padding="same"))

    model.add(Activation('softmax'))

    print(model.summary())
    return model