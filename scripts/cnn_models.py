""" Here we wrap the cnn model in a class to provide a simple interface
    used to train, cross validate, evaluate, save and load the model 
    and show predictions"""

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
from evaluate import evaluate_predictions, display_prediction, plot_history 
from functools import reduce
import json
from types import SimpleNamespace 

from preprocessing import *
from postprocessing import *
import imageio
from mask_to_submission import *

class CnnModel:
    def __init__(self, model_n=0, model_path=""):
        """ Initilizes a model.
            model_n: indicates which model has to be created
            model_path: here we save the best weight obtained during the training. If save() is called
                the model, the current weights and its history will be saved in this path.
            """
        # here the list of the functions that create a model
        models = [model1, model2,additional_conv_layer_model,max_pooling_model,leaky_relu_model,decreased_dropout_model,many_filters_model, model_leakyrelu_maxpooling, model_relu_maxpooling] 
        self.model_names = ["model1","model2","additional_conv_layer","max_pooling","leaky_relu_model","decreased_dropout","many_filters", "model_leakyrelu_maxpooling", "model_relu_maxpooling"]
        self.model_idx = model_n
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

    def predict_and_export(self):
        threshold = 0.5
        filename_list = []
        for batch_idx in range(1):
            images = load_images_to_predict(batch_idx*5,batch_idx*5+5)
            predictions = self.model.predict(images)
            
#             img_matrix = np.zeros([predictions.shape[0],predictions.shape[1],predictions.shape[2]])
            
#             print(img_matrix.shape)
            for i in range(predictions.shape[0]):
#                 for row in range(predictions.shape[1]):
#                     for col in range (predictions.shape[2]):
#                         if  predictions[i][row][col][1]>threshold :
#                             img_matrix[i][row][col] = 1.0
#                         else:
#                             img_matrix[i][row][col] = 0.0
               
                img_idx = batch_idx*5+i+1
                filename = "../dataset/test_set_images/test_"+str(img_idx)+"/gt"+str(img_idx)
                ext = ".jpg"
                pred = prediction_to_class(predictions[i], threshold = threshold) 
                imageio.imwrite(filename+ext, pred)
                filename_list.append(filename+ext)
        
        masks_to_submission("submission.csv",filename_list)
        
    def predict_augmented_and_export(self):
        threshold = 0.5
        filename_list = []
        for batch_idx in range(1):
            images = load_images_to_predict(batch_idx*5,batch_idx*5+5)
            predictions = self.model.predict(images)
            for i in range(predictions.shape[0]):              
                img_idx = batch_idx*5+i+1
                filename = "../dataset/test_set_images/test_"+str(img_idx)+"/gt"+str(img_idx)
                ext = ".jpg"
                pred = prediction_to_class(predictions[i], threshold = threshold) 
                imageio.imwrite(filename+ext, pred)
                filename_list.append(filename+ext)
        
        masks_to_submission("submission.csv",filename_list)
    
    def predict(self, x, batch_size=None, verbose=0, steps=None):
        return self.model.predict(x)#,batch_size = batch_size, verbose = verbose, steps = steps)
        
    def predict_augmented(self, imgs, n_rotations=4, verbose=True):
        """ Predict the probability of each patch of the input image.
        Does so for 2*n_rotations times (augmenting the image) and averaging the predictions. 
        img.shape = (#images, H, W, 3)
        returned shape = (#images, H/n, W/n), floats indicating the road probabilities
        verbose: True => print the status. False => print nothing"""
        
        # flip + rotate by 360/n_rotations * k (k=1, ..., n_rotations) => 2*n_rotations transformations (= #predictions)
        num_images = imgs.shape[0]
        rot_interval = int(360/n_rotations) 
        prediction_shape = self.predict(imgs[:1])[0].shape # (I just get shape of the prediction)
        predictions = np.zeros(np.append(num_images, prediction_shape)) # here we sum all the predictions 
        for i in range(num_images):
            print("Predicting image", i) if verbose else None
            for flip in [False, True]:
                print("\tFlipping image:", flip) if verbose else None
                img_flipped = np.flip(imgs[i], axis=1) if flip else imgs[i]
                for degrees in range(0, 360, rot_interval):
                    print("\t\tRotate by", degrees, "degrees and predict") if verbose else None
                    img_aug = rotate_image(img_flipped, degrees)
                    # predict, the image 
                    curr_pred = self.predict(np.array([img_aug]))[0]
                    # flip and rotate back, and sum
                    # when rotating back, I may obtain a bigger image of which I have to take just the one at the center
                    curr_pred = take_image_at_center(rotate_image(curr_pred, -degrees), prediction_shape) 
                    curr_pred = np.flip(curr_pred, axis=1) if flip else curr_pred
                    predictions[i] += curr_pred

        n_transformations = 2*n_rotations
        predictions /=  n_transformations
        return predictions
    
    def name(self):
        return self.model_names[self.model_idx]
    
    def summary(self):
        """ Print the layers of the model. """
        if hasattr(self, 'name'):
            print(self.name())
        print(self.model.summary())
        
    def compile(self):
        """ Compiles the model: define the learning process """
#         def softmax_categorical_crossentropy(y_true, y_pred):
#             """
#             Uses categorical cross-entropy from logits in order to improve numerical stability.
#             This is especially useful for TensorFlow (less useful for Theano).
#             """
#             return K.categorical_crossentropy(y_pred, y_true, from_logits=True)

#         opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
#         self.model.compile(loss=softmax_categorical_crossentropy,
#                       optimizer=opt,
#                       metrics=['accuracy'])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        
    def train(self, train, test=None, num_epochs=10, batch_size=5, monitor='val_loss', save_history=True):
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
            filepath = self.model_path+"\\"+str('{0:%Y-%m-%d_%H%M%S}'.format(datetime.now()))+"_best-weights"+self.model_names[self.model_idx]+".hdf5" 
            checkpoint = callbacks.ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

        # we compute steps_per_epoch so that all the images are used for every epoch
        steps_per_epoch = np.ceil(train.X.shape[0]/batch_size).astype("int")

        try:
            hist = self.model.fit_generator(
                batches_generator(train.X, train.Y, batch_size = batch_size),
                steps_per_epoch=steps_per_epoch, # number of batches to train on at each epoch
                epochs=num_epochs,
                verbose=1,
                validation_data=(None if test==None else (test.X, np_utils.to_categorical(test.Y, 2))),
                callbacks=callbacks_list
            )
            if save_history == True:
                # update the history (apped the new losses and accuracies so to plot them)
                for hist_key in self.history:
                    if hist_key not in hist.history:
                        # if you don't have the val_loss or the val_acc, set them to a list of 0s
                        values = [0 for i in range(num_epochs)]
                    else:
                        values = hist.history[hist_key]
                    self.history[hist_key] = self.history[hist_key] + values 

            return hist.history  # return the history of just this run

        except KeyboardInterrupt:
            # if the user stops the training process keep the model 
            print("Interrupt")
            pass
        
    def cross_validation(self, set_, num_epochs=10, batch_size=5):
        """
        -set_: set_.X are all the input images, set_.Y are the respective groundtruths
        -num_epochs: number of epoch each one of the 4 trains should last
        -batch_size: number of images in each batch
        Splits the set_ in 4 folds and train one time on each one of them and starting 
        from scratch every time (reset the weights). No best weights are stored (this
        fucntion should be used just to compare different models). 
        The obtained 'history' is an average of the histories of the four iterations. 
        This function returns and also stores a file in the model's folder with all the 
        parameters used and the obtained results (e.g. average F1 score, average history,
        number of epochs, the batch size, ...).
        The results will allow to compare the models.
            """

        # divide the data in 4 folds
        nfolds = 4
        histories = [] # here I append the 4 histories which will be then averaged
        F1_scores = []
        kf = KFold(n_splits=nfolds)
        train = SimpleNamespace()
        test = SimpleNamespace()
        for train_indices, test_indices in kf.split(range(set_.X.shape[0])): # split the indices
            # mix the weights (random reinitialize)
            weights = self.model.get_weights()
            weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
            self.model.set_weights(weights)
        
            # get train and test
            train.X = set_.X[train_indices]
            train.Y = set_.Y[train_indices]
            test.X = set_.X[test_indices]
            test.Y = set_.Y[test_indices]
            
            # fit the model
            h = self.train(train, test, num_epochs=num_epochs, batch_size=batch_size, monitor='', save_history=False)
            
            # store the history and the f1 score
            histories.append(h)
            # compute F1 score of the road labels
            pred = self.model.predict_classes(test.X).flatten()
            true = test.Y.flatten()
            f1 = f1_score(true, pred, pos_label=1, average='binary', sample_weight=None)
            F1_scores.append(f1)
        
        # compute the mean history
        mean_history = {}
        keys = histories[0].keys()
        for key in keys:
            mean_history[key] = np.zeros(len(histories[0][key]))
        for key in keys:
            for h in histories:
                mean_history[key] += np.array(h[key])
            mean_history[key] = (mean_history[key]/len(histories)).tolist()
            
        results = {
            "n_images": set_.X.shape[0],
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "histories": histories,
            "history_mean": mean_history,
            "f1_scores": F1_scores,
            "f1_mean": np.mean(F1_scores), # estimate of the expected f1 score
            "f1_std": np.std(F1_scores)    # how much we can trust that estimate
        }
        
        # store the results
        filepath = self.model_path+"/cross_validation_" + str('{0:%Y-%m-%d_%H%M%S}'.format(datetime.now())) +self.model_names[self.model_idx]+".json"
        with open(filepath, "w") as json_file:
            json_file.write(json.dumps(results, indent=4))
        
        return results
            
    def plot_history(self, last_epochs=-1):
        """ Plot the history of the model (loss and accuracies obtained duting the training).
            If last_epochs!=-1 then plot only the last give number of epochs
        """
        plot_history(self.history,self.name(), last_epochs=last_epochs)    
            
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
        with open(self.model_path+"/model-"+self.name()+".json", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5 and save them to file
        self.model.save_weights(self.model_path+"/weights-"+self.name()+".h5")
        
        # serialize the history to json and store it
        with open(self.model_path + '/history-'+self.name()+'.json', 'w') as file:
            file.write(json.dumps(self.history, indent=4))
        
        print("Saved model to disk")

    def load(self):
        """ Loads model, weights and history stored in 'dirpath' """
        
        # load json and create model
        json_file = open(self.model_path+'/model-'+self.name()+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights(self.model_path + "/weights-"+self.name()+".h5")
        
        # reset the history (note: we are not storing the history)
        with open(self.model_path + '/history-'+self.name()+'.json', 'r') as file:
            self.history = json.loads(file.read())
        
        print("Loaded model from disk")    
        
    def load_weights(self, file):
        """ Load  from the given file the weights to the current model """
        self.model.load_weights(self.model_path + "/" + file)
        
   
        
        
def image_generators(X, Y):
    """ 
        Given the input images (X), their groundtruth (Y) returns two generators. 
        Both the generators return two images per time (respectively the input 
        image and the relative groundtruth). 
        Both the generators augment the images by randomly applying horizontal flip, 
        zoom (to improve prediction of wider and thinner roads) and filling the points 
        outside the boundaries by mirroring the images. 
        The two generators differ in the rotation: the first generator rotates the images
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
    
    batch_size = 1 # take one image per time
    Y = np.expand_dims(Y, axis=3) # "wrap" each pixel in a numpy array

    ### 1. generator with rotations of 90*k 
    # use a seed so to apply the same transformation to both the input image (X) and the 
    # groundtruth (Y)
    seed = 5
    
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
        for x, y in generator_no_rot:
            k = np.random.randint(4)
            x_rot = np.rot90(x[0], k=k, axes=(0, 1))
            y_rot = np.rot90(y[0], k=k, axes=(0, 1))
            yield x_rot, y_rot
    
     ### 2. generator with any rotation
    seed = 4
    data_gen_args["rotation_range"] = 360 # add the rotation parameter
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # combine generators into one which yields image and masks
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)
    generator_rot360_ = zip(image_generator, mask_generator)
    
    def generator_rot360():
        # just change the shape of the output (instead of returning a batch
        # with only one image retutn that image)
        for x, y in generator_rot360_:
            yield x[0], y[0]
        
    return generator_rot90k(), generator_rot360()

def batches_generator(X, Y, batch_size=4):
    """ Combine the two generators obtainef from image_generators to obtain 
        the batch generator used during training.
        X, Y: input and output images
        prob_gen1: probability of using gen1 to generate the next batch. 
        batch_size: number of images in each batch. """
    
    gen1, gen2 = image_generators(X, Y) 
    prob_gen1 = 0.8
    
    x, y = next(gen1) # just to take the shape of x and y
    while 1:
        # generate the batch
        batch_x = np.zeros((np.append(batch_size, x.shape))).astype('float32')
        batch_y = np.zeros((np.append(batch_size, y.shape))).astype('float32')
        
        for i in range(batch_size):
            if np.random.rand()<prob_gen1:
                # then use gen1
                bx, by = next(gen1)
#                 print("gen1: generated images: ", bx.shape, by.shape)
                batch_x[i], batch_y[i] = bx, by
            else:
                bx, by = next(gen2)
#                 print("gen2: generated images: ", bx.shape, by.shape)
                batch_x[i], batch_y[i] = bx, by
                
#         print("Generated x and y batch of sizes: ", batch_x.shape, batch_y.shape, batch_y.dtype, batch_x.dtype)
        yield batch_x, np_utils.to_categorical(batch_y, 2).astype('float32') # convert Y to categorical (each pixel is either [1, 0] or [0, 1])
            
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
def additional_conv_layer_model():
    # with relu from keras import backend as K
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
    
    # layer 9
    model.add(
        Conv2D(2, (5, 5), 
               activation='relu',
               padding="same"))
    
    model.add(Activation('softmax'))

    return model

def many_filters_model():
    # with relu from keras import backend as K
    nclasses = 2
    model = Sequential()
    pool_size = (2, 2)

    # layer 1
    model.add(
        Conv2D(64, (11, 11), 
               activation='relu',
               padding="same", 
               strides=(2, 2),
               input_shape=(None, None, 3)))
    model.add(Dropout(0.25)) 

    # layer 2
    model.add(
        Conv2D(64, (5, 5),
               activation='relu',
               strides=(2, 2),
               padding="same"
              ))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

    # layer 2
    model.add(
        Conv2D(64, (5, 5),
               activation='relu',
               strides=(2, 2),
               padding="same"
              ))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

    # layer 4
    model.add(
        Conv2D(2, (5, 5),
               activation='relu',
               strides=(2, 2),
               padding="same"))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

    
    model.add(Activation('softmax'))

    return model

def max_pooling_model():
    # with relu from keras import backend as K
    nclasses = 2
    model = Sequential()
    pool_size = (2, 2)

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
               strides=(1, 1),
               padding="same"
              ))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

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
               strides=(1, 1),
               padding="same"))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

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
               strides=(1, 1),
               padding="same"))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))

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

def leaky_relu_model():
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

    # later 3
    model.add(
        Conv2D(48, (5, 5), 
               padding="same")) 
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25)) 

    # layer 4
    model.add(
        Conv2D(48, (5, 5),
               strides=(2, 2),
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    # layer 5
    model.add(
        Conv2D(48, (5, 5), 
               padding="same")) 
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25)) 

    # layer 6
    model.add(
        Conv2D(64, (5, 5), 
               strides=(2, 2),
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    # layer 7
    model.add(
        Conv2D(64, (5, 5), 
               padding="same")) 
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25)) 

    # layer 8
    model.add(
        Conv2D(2, (5, 5), 
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Activation('softmax'))
    return model


def decreased_dropout_model():
# with relu from keras import backend as K
    nclasses = 2
    model = Sequential()

    # layer 1
    model.add(
        Conv2D(32, (11, 11), 
               activation='relu',
               padding="same", 
               input_shape=(None, None, 3)))
    model.add(Dropout(0.1)) 

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
    model.add(Dropout(0.1)) 

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
    model.add(Dropout(0.1)) 

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
    model.add(Dropout(0.1)) 

    # layer 8
    model.add(
        Conv2D(2, (5, 5), 
               activation='relu',
               padding="same"))

    model.add(Activation('softmax'))

    return model

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
    # with relu from keras import backend as K
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

def model_leakyrelu_maxpooling():
    # with relu from keras import backend as K
    nclasses = 2
    model = Sequential()
    pool_size = (2,2)
    
    # layer 1
    model.add(
        Conv2D(48, (7, 7), 
               padding="same", 
               input_shape=(None, None, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20)) 

    # layer 2
    model.add(
        Conv2D(64, (5, 5),
               padding="same"
              ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20))

    # later 3
    model.add(
        Conv2D(128, (5, 5), 
               padding="same")) 
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20)) 

    # layer 4
    model.add(
        Conv2D(256, (5, 5), 
               padding="same")) 
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.20)) 

    # layer 5
    model.add(
        Conv2D(2, (5, 5), 
               padding="same"))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Activation('softmax'))

    return model

def model_relu_maxpooling():
    # with relu from keras import backend as K
    nclasses = 2
    model = Sequential()
    pool_size = (2,2)
    
    # layer 1
    model.add(
        Conv2D(48, (7, 7), 
               padding="same", 
               activation='relu',
               input_shape=(None, None, 3)))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20)) 

    # layer 2
    model.add(
        Conv2D(64, (5, 5),
               padding="same",
               activation='relu',
              ))
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20))

    # later 3
    model.add(
        Conv2D(128, (5, 5), 
               activation='relu',
               padding="same")) 
    model.add(MaxPooling2D(padding="same",pool_size=pool_size))
    model.add(Dropout(0.20)) 

    # layer 4
    model.add(
        Conv2D(256, (5, 5), 
               activation='relu',
               padding="same")) 
    model.add(Dropout(0.20)) 

    # layer 5
    model.add(
        Conv2D(2, (5, 5), 
               activation='relu',
               padding="same"))

    model.add(Activation('softmax'))

    return model