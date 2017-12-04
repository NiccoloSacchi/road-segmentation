""" All functions used to evaluate the predictions and show plots """

from sklearn.metrics import classification_report,confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from tf_aerial_images import make_img_overlay

# TODO fix this function
def display_prediction(model, img, ax=None):
    """ Display predictions on the image. 
        img: original image. shape=(W, H, 3)
    """
    Z = model.predict_classes(np.array([img]))[0]
    
    h, w = img.shape[0], img.shape[1] # 608, 608 for test set

    predicted_gt = predictions_to_img(Z, (h, w))
    new_img = make_img_overlay(img, predicted_gt)

    if ax != None:
        ax.imshow(new_img)
        return
    
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(new_img)
    return

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

def plot_history(hist):
    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(len(hist.history['loss']))

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
    
def show_layer_output(model, image, layer_num, filename=""):
    """ Use this function to plot the output (activations) of a layer. 
        model: the model you trained
        images: the image to be fed to the model
        layer_num: layer whose output you are interested in
        filename: if different from "" the image is also stored
    """
    
    # Function to get the activations of an intermediate layer
    def get_featuremaps(model, layer_idx, X_batch):
        get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
        activations = get_activations([X_batch,0])
        return activations

    activations = get_featuremaps(model, int(layer_num), [image])

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
    
    
def evaluate_model(model, X, Y):
    """ Evaluate the model oh the given data """
    pred = model.predict_classes(X).flatten()
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