""" Here we put all our cnn models """

# import 'Sequential' is a linear stack of neural network layers. Will be used to build the feed-forward CNN
from keras.models import Sequential 
# import the "core" layers from Keras (these are the most common layers)
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
# import the convolutional layers that will help us efficiently train on image data
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
    
from keras import callbacks

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from preprocessing import rand_augment_images
from datetime import datetime
from sklearn.metrics import classification_report,confusion_matrix
from tf_aerial_images import make_img_overlay

class CnnModel:
    def __init__(self, model_n=0, model_path=""):
        """ Initilizes a model.
            model_n: indicates which model has to be created
            model_path: here we save the best weight obtained during the training. If save() is called
                the model, the current weights and its history will be saved in this path.
            """
        models = [model1, model2] # here the list of the functions that create a model
        self.model = models[model_n]() 
        self.history = {
            'acc': [],
            'loss': [],
            'val_acc': [],
            'val_loss': []
        }
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path=model_path
        
    def summary(self):
        """ Print the layers of the model. """
        print(self.model.summary())
        
    def compile(self):
        """ Compiles the model: define the learning process """
        # TODO here we should define our loss function        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    
    def train(self, train, test, num_epochs=10, batch_size=5):
        """ Train the model with the train set (train.X and train.Y).
            It also computes the validation error at each epoch on the train set.
        """
        
        # use a callback to save the best model (best model = the one with the best 'monitor' variable)
        # could also add "-{epoch:03d}-{loss:.4f}-{acc:.4f}" to the name
        filepath = self.model_path+"/"+str('{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.now()))+"_best-weights.hdf5" 
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # we compute steps_per_epoch so that all the images are used for every epoch
        steps_per_epoch = np.ceil(train.X.shape[0]/batch_size).astype("int")

        def batches_generator(batch_size, train):
            """ This function returns a generator to generate an infinite number of batches
                from train.imgs and train.gt_imgs which are randomly augmented (randomly 
                flipped and rotated) every time a batch is created.
            """
            indices = np.arange(train.X.shape[0])
            while 1:
                # shuffle the images
                shuffled = np.random.permutation(indices)
                if len(shuffled)%batch_size != 0:
                    # pad shuffled so that it is a multiple of batch_size
                    remaining = len(shuffled)%batch_size
                    shuffled = np.concatenate((shuffled, np.random.choice(indices, size=remaining, replace=False)))

                for i in range(0, len(shuffled), batch_size):
                    # take the images and the respective groundtruth for the current batch
                    batch_images = shuffled[i:i+batch_size]
                    # augment them randomly (flip + rotation)
                    inputs, targets = rand_augment_images(train.X[batch_images], train.Y[batch_images])

        #             print("\nbatch generated: ", inputs.shape, targets.shape)
                    yield (inputs, targets)

        try:
            hist = self.model.fit_generator(batches_generator(batch_size, train),
                                      steps_per_epoch=steps_per_epoch, # number of batches to train on at each epoch
                                      epochs=num_epochs,
                                      verbose=1,
                                      validation_data=(test.X, test.Y),
                                      callbacks=callbacks_list
                                     )
            # update the history (apped the new losses and accuracies so to plot them)
            for hist_key in hist.history:
                if hist_key not in self.history:
                    self.history = []
                self.history[hist_key] = self.history[hist_key] + hist.history[hist_key]
                
        except KeyboardInterrupt:
            # if the user stops the training process keep the model 
            pass
        
    def plot_history(self, last_epochs=-1):
        """ Plot the history of the model (loss and accuracies obtained duting the training).
            If last_epochs!=-1 then plot only the last give number of epochs
        """
        num_epochs = len(self.history['loss']) if last_epochs==-1 else last_epochs
        # visualizing losses and accuracy
        train_loss=self.history['loss'][-num_epochs:]
        val_loss=self.history['val_loss'][-num_epochs:]
        train_acc=self.history['acc'][-num_epochs:]
        val_acc=self.history['val_acc'][-num_epochs:]
        xc=range(len(self.history['loss']))[-num_epochs:]

        plt.figure(1,figsize=(7,5))
        plt.plot(xc,train_loss)
        plt.plot(xc,val_loss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('train_loss vs val_loss')
        plt.grid(True)
        plt.legend(['train','val'])

        plt.figure(2,figsize=(7,5))
        plt.plot(xc,train_acc)
        plt.plot(xc,val_acc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.ylim([0, 1])
        plt.title('train_acc vs val_acc')
        plt.grid(True)
        plt.legend(['train','val'],loc=4)
            
    def show_layer_output(self, image, layer_num, filename=""):
        """ Use this function to plot the output (activations) of a layer. 
            model: the model you trained
            images: the image to be fed to the model
            layer_num: layer whose output you are interested in
            filename: if different from "" the image is also stored
        """

        # Function to get the activations of an intermediate layer
        def get_featuremaps(layer_idx, X_batch):
            get_activations = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[layer_idx].output,])
            activations = get_activations([X_batch,0])
            return activations

        activations = get_featuremaps(int(layer_num), [image])

        feature_maps = activations[0][0]      
        print("feature_maps (output of layer " + str(layer_num) + "):", feature_maps.shape)

        # show all the filters
        num_of_featuremaps=feature_maps.shape[2]
        fig=plt.figure(figsize=(16,16))
        plt.title("featuremaps of layer {}".format(layer_num))
        subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
        for i in range(int(num_of_featuremaps)):
            ax = fig.add_subplot(subplot_num, subplot_num, i+1)
            ax.imshow(feature_maps[:,:,i])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
        plt.show()

        if filename != "":
            fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')
        # you should be able to see that the cnn is keeping less information while the image goes through (keep only the
        # important features)
        
    def evaluate_model(self, X, Y):
        """ Evaluate the model oh the given data """
        pred = self.model.predict_classes(X).flatten()
        true = np.argmax(Y, axis=len(Y.shape)-1).flatten()

        target_names = ['0 (background)', '1 (road)']

        # print precision, recall, F1 score
        print(classification_report(true, pred, target_names=target_names))

        # Plotting the confusion matrix
        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.apply_along_axis(lambda a: (a // 0.01)*0.01, 1, cm)
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            thresh = cm.max() / 1.2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        # Compute confusion matrix
        cnf_matrix = (confusion_matrix(true, pred))

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix')

        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                             title='Normalized confusion matrix')
        
    def display_prediction(self, img, ax=None):
        """ Display predictions on the image. 
            img: original image. shape=(W, H, 3)
        """
        def predictions_to_img(Y, size):
            """ From the matrix with the predicted patches build an image of the given size 
                size=(height, width). """
            # how many times we have to repeat h
            h_times = size[0]/Y.shape[0]
            # how many times we have to repeat w
            w_times = size[1]/Y.shape[1]

            if h_times%1 != 0 or w_times%1 != 0:
                print("Impossible to build an image of the given size")
                return None

            gt_img = np.repeat(np.repeat(Y, h_times, axis=0), w_times, axis=1)
            return gt_img

        Z = self.model.predict_classes(np.array([img]))[0]

        h, w = img.shape[0], img.shape[1] # 608, 608 for test set

        predicted_gt = predictions_to_img(Z, (h, w))
        new_img = make_img_overlay(img, predicted_gt)

        if ax != None:
            ax.imshow(new_img)
            return

        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(new_img)
        return
    
    def save(self):
        """ Creates the directory 'dirpath' and save both the model and the weights.
            N.B. use a different 'dirpath' for each model you want to keep
        """    
    
        # serialize model to JSON and save it to file
        model_json = self.model.to_json()
        with open(self.model_path+"/model.json", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5 and save them to file
        self.model.save_weights(self.model_path+"/weights.h5")
        
        # serialize the history to json and store it
        np.save(self.model_path + '/history.npy', self.history) 
        
        print("Saved model to disk")

    def load(self):
        """ Loads model, weights and history stored in 'dirpath' """
        
        # load json and create model
        json_file = open(self.model_path+'/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights(self.model_path + "/weights.h5")
        
        # reset the history (note: we are not storing the history)
        self.history = read_dictionary = np.load(self.model_path + '/history.npy').item()
        
        print("Loaded model from disk")    
        
    def load_weights(self, file):
        """ Load just the weights from the given file """
        self.model.load_weights(self.model_path + "/" + file)
        
        
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

    return model

def model2():
    # with relufrom keras import backend as K
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

    return model