""" Here we wrap the cnn model in a class to provide a simple interface
    used to train, test, save, load the model and show predictions"""

# import 'Sequential' is a linear stack of neural network layers. Will be used to build the feed-forward CNN
from keras.models import Sequential 
# import the "core" layers from Keras (these are the most common layers)
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
# import the convolutional layers that will help us efficiently train on image data
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras import callbacks
from keras.utils import np_utils    

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
from sklearn.metrics import classification_report,confusion_matrix
from tf_aerial_images import make_img_overlay
from evaluate import evaluate_predictions, display_prediction
from matplotlib.ticker import MaxNLocator

class CnnModel:
    def __init__(self, model_n=0, model_path=""):
        """ Initilizes a model.
            model_n: indicates which model has to be created
            model_path: here we save the best weight obtained during the training. If save() is called
                the model, the current weights and its history will be saved in this path.
            """
        models = [model1, model2, model3] # here the list of the functions that create a model
        self.model = models[model_n]() 
        self.history = {
            'acc': np.array([]),
            'loss': np.array([]),
            'val_acc': np.array([]),
            'val_loss': np.array([])
        }
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path=model_path
        
    def summary(self):
        """ Print the layers of the model. """
        print(self.model.summary())
        
    def compile(self):
        """ Compiles the model: define the learning process """
        # TODO here we should define our loss function (not with the new generation)     
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        
    def train(self, train, test=None, num_epochs=10, batch_size=5, monitor='val_loss'):
        """ Train the model with the train set (train.X and train.Y).
            -train: a structure where train.X are the input images and train.Y are the true
                predictions
            -train: if !=None, we also validate the model at each epoch on this test set.
            -num_epochs: number of epoch to train the model
            -batch_size: number of images in each batch
            -monitor: {'loss', 'acc', 'val_loss', 'val_acc', ''}, indicate which parameter we should 
                monitor to decide which is the best model. During the training we store the 
                current weights every time the 'monitor' parameter improves.
                N.B. If you don't pass any 'test' set then monitor parameter should be in {'loss', 'acc'}
        """
        self.compile()
        
        # use a callback to save the best model (best model = the one with the best 'monitor' variable)
        # could also add "-{epoch:03d}-{loss:.4f}-{acc:.4f}" to the name
        callbacks_list = []
        if monitor != '':
            filepath = self.model_path+"/"+str('{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.now()))+"_best-weights.hdf5" 
            checkpoint = callbacks.ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

        # we compute steps_per_epoch so that all the images are used for every epoch
        steps_per_epoch = np.ceil(train.X.shape[0]/batch_size).astype("int")
        
        generator_rot90k, generator_rot360 = image_generators(train.X, train.Y, batch_size = batch_size)
        batches_generator = combine_generators(generator_rot90k, generator_rot360, prob_gen1=0.8)

        try:
            hist = self.model.fit_generator(
                batches_generator,
                steps_per_epoch=steps_per_epoch, # number of batches to train on at each epoch
                epochs=num_epochs,
                verbose=1,
                validation_data=(None if test==None else (test.X, np_utils.to_categorical(test.Y, 2))),
                callbacks=callbacks_list
            )
            # update the history (apped the new losses and accuracies so to plot them)
            for hist_key in self.history:
                if hist_key not in hist.history:
                    # if you don't have the val_loss or the val_acc, set them to a list of 0s
                    values = np.array([0 for i in range(num_epochs)])
                else:
                    values = np.array(hist.history[hist_key])
                self.history[hist_key] = np.append(nself.history[hist_key], values) 
             
            return hist.history  # return the history of just this run
        
        except KeyboardInterrupt:
            # if the user stops the training process keep the model 
            print("Interrupt")
            pass
        
#     def cross_validation(self, set_, num_epochs=10, batch_size=5):
#         """ Is similar to train() but splits the set_ in 4 fold and train one time on each 
#             one of them and starting from scratch every time (reset the weights). No best 
#             weights are stored. 
#             The obtained 'history' is an averas of the four iterations. 
#             set_: set_.X are all the input images, set_.Y are the respective groundtruths 
#             num_epochs: number of epoch each one of the 4 train should last
#             batch_size: number of images in each batch
            
#             We store a file in the model's folder with: the average F1 score obtained, the 
#                 number of epochs, the batch size and the average history. These paarmeters 
#                 will allow to compare the models.
#             """
            
#         # divide the data in 4 folds
#         nfolds = 4
#         histories = [] # here I append the 4 histories which will be then averaged
#         F1_scores = []
#         kf = KFold(n_splits=nfolds)
#         train = SimpleNamespace()
#         test = SimpleNamespace()
#         for train_indices, test_indices in kf.split(range(set_.X.shape[0])): # split the indices
#             train.X = set_.X[train_indices]
#             train.Y = set_.Y[train_indices]
#             test.X = set_.X[test_indices]
#             test.Y = set_.Y[test_indices]
#             h = self.train(train, test, num_epochs=num_epochs, batch_size=batch_size, monitor='')
#             histories.append(h)
            
#             # compute F1 score of the road labels
#             pred = model.model.predict_classes(test.X).flatten()
#             true = test.Y.flatten()
#             f1 = f1_score(true, pred, pos_label=1, average='binary', sample_weight=None)
#             F1_scores.append(f1)
         
#         return histories, history_average, F1_scores, F1_average
            
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

        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches((7, 14))
        axs[0].plot(xc,train_loss)
        axs[0].plot(xc,val_loss)
        axs[0].set_xlabel('num of Epochs')
#         plt.ticklabel_format(style='plain')
        axs[0].set_ylabel('loss')
        axs[0].set_title('train_loss vs val_loss')
        axs[0].grid(True)
        axs[0].legend(['train','val'])
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        axs[1].plot(xc,train_acc)
        axs[1].plot(xc,val_acc)
        axs[1].set_xlabel('num of Epochs')
        axs[1].set_ylabel('accuracy')
        axs[1].set_ylim([0, 1])
        axs[1].set_title('train_acc vs val_acc')
        axs[1].grid(True)
        axs[1].legend(['train','val'],loc=4)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            
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
        true = Y.flatten()
#         true = np.argmax(Y, axis=len(Y.shape)-1).flatten()
        evaluate_predictions(pred, true)
        
    def display_prediction(self, img, ax=None):
        """ Display predictions on the image. 
            img: original image. shape=(W, H, 3)
        """
        prediction = self.model.predict_classes(np.array([img]))[0]
        display_prediction(img, prediction, ax=None)
    
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
        """ Load  from the given file the weights to the current model """
        self.model.load_weights(self.model_path + "/" + file)
        
def image_generators(X, Y, batch_size = 32):
    """ 
        Given the input images (X), their groundtruth (Y) and the size of the batch
        returns two generators. 
        Both of the generators return two sets of images (respectively with the input 
        images and the relative groundtruth) of size = batch_size. 
        Both the generators augment the images by randomly applying horizontal flip, 
        zoom (to improve prediction of wider and thinner roads) and filling the points 
        outside the boundaries by mirroring the images. 
        However, they differ in the rotation: the first generator rotates the images
        of degrees 90*k (where k=0,1,2,3), the second generator rotates the images 
        of any degree. """
    # TODO does it makes sense to shift (probably not)?
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    # TODO do we want to normalize the input? (requires the .fit call to compute mean and std)
    # normalize with featurewise_center and featurewise_std_normalization
    #     samplewise_center=True,
    #     samplewise_std_normalization=True
        # # Provide the same seed and keyword arguments to the fit and flow methods
        # image_datagen.fit(X, augment=True, seed=seed)
        # mask_datagen.fit(Y, augment=True, seed=seed)
    
    Y = np.expand_dims(Y, axis=3) # "wrap" each pixel in a numpy array
 
    # use a seed so to apply the same transformation to both the input image (X) and the 
    # groundtruth (Y)
    seed = 1

    ### 1. generator with rotations of 90*k 
    data_gen_args = dict(    
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="reflect"
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # combine generators into one which yields image and masks
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)
    
    generator_no_rot = zip(image_generator, mask_generator)
    
    def generator_rot90k():
#         print("entered generator_rot90k (should happen just one time)")
        # take the generator with no rotation and apply a rotation of 90*k
        for x_batch, y_batch in generator_no_rot:
            x_batch_rot = np.zeros(x_batch.shape) 
            y_batch_rot = np.zeros(y_batch.shape)
            for i in range(x_batch.shape[0]):
                k = np.random.randint(4)
                x_batch_rot[i] = np.rot90(x_batch[i], k=k, axes=(0, 1))
                y_batch_rot[i] = np.rot90(y_batch[i], k=k, axes=(0, 1))
            yield x_batch_rot, y_batch_rot
    
     ### 2. generator with any rotation
    data_gen_args["rotation_range"] = 360 # add the rotation parameter
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # combine generators into one which yields image and masks
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)
    generator_rot360 = zip(image_generator, mask_generator)
    
    return generator_rot90k(), generator_rot360

def combine_generators(gen1, gen2, prob_gen1=0.8):
    """ Combine two generators to obtain the batch generator used during training.
        prob_gen1: probability of using gen1 to generate the next batch. """

    while 1:
        if np.random.rand()<prob_gen1:
            # then use gen1
            batch_x, batch_y = next(gen1)
        else:
            batch_x, batch_y = next(gen2)
        print("\nGenerated x and y batch of sizes: ", batch_x.shape, batch_y.shape)
        yield batch_x, np_utils.to_categorical(batch_y, 2) # cobvert Y to categorical (each pixel is either [1, 0] or [0, 1])
            
# def batches_generator(batch_size, train):
#     """ This function returns a generator to generate an infinite number of batches
#         from train.imgs and train.gt_imgs which are randomly augmented (randomly 
#         flipped and rotated) every time a batch is created.
#     """
#     indices = np.arange(train.X.shape[0])
#     while 1:
#         # shuffle the images
#         shuffled = np.random.permutation(indices)
#         if len(shuffled)%batch_size != 0:
#             # pad shuffled so that it is a multiple of batch_size
#             remaining = len(shuffled)%batch_size
#             shuffled = np.concatenate((shuffled, np.random.choice(indices, size=remaining, replace=False)))

#         for i in range(0, len(shuffled), batch_size):
#             # take the images and the respective groundtruth for the current batch
#             batch_images = shuffled[i:i+batch_size]
#             # augment them randomly (flip + rotation)
#             inputs, targets = rand_augment_images(train.X[batch_images], train.Y[batch_images])

# #             print("\nbatch generated: ", inputs.shape, targets.shape)
#             yield (inputs, targets)

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