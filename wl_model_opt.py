from keras.utils import normalize
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from multi_unet import multi_model, jacard
import rasterio as rio
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K

# os.environ['SM_FRAMEWORK']=

#%env SM_FRAMEWORK=tf.keras
import segmentation_models as sm
import optuna







####test : i can actually write this myself easily enough, i know exactly what it needs to do.


from keras import backend as K
import numpy as np



from scipy.ndimage import distance_transform_edt as distance





##




###







def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).reshape(y_true.shape).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)



#end test












def objective(trial):



    #794
    n_patches=795

    n_classes=6

    metrics=['accuracy', jacard]




    #get images
    train_images=[]
    for i in range(n_patches):
        with rio.open('/home/willh/projects/def-rriordan-ab/willh/natural_assets_models/wlpatches/patch_%s.tif'%i) as data:

            train_images.append(np.asarray([data.read(1),data.read(2),data.read(3),data.read(4),data.read(5),data.read(6),data.read(7),data.read(8)]))

    train_images2=np.asarray(train_images)[:,:,:,:]



    #
    # train_images=np.asarray(train_images)[:,np.r_[0:2, 3:4, 7:8],:,:]

    train_images=np.asarray(train_images)[:,np.r_[0:8],:,:]
    #
    # use all bands
    # train_images=train_images2


    #frm now on, 0 = blue , 1 = green , 2 = red , 3 = NI

    #putting into tensor format with different channel data stored in most-nested vector like (number patches, height, width, channels)

    #test
    train_images2=train_images.swapaxes(1,3)
    train_images2=train_images.swapaxes(1,2)
    #


    train_images=train_images.swapaxes(1,3)
    train_images=train_images.swapaxes(1,2)




    print (train_images.shape)




    #get masks
    train_masks=[]
    for i in range(n_patches):
        with rio.open('/home/willh/projects/def-rriordan-ab/willh/natural_assets_models/wlmasks/mask_%s.tif'%i) as data:
            train_masks.append((data.read(1)))

    train_masks=np.asarray(train_masks)
    print (train_masks.shape)

    train_masks_cleaned=[]
    train_images_cleaned=[]
    train_images_cleaned2=[]

    #remove cases with gaps in labelled pixels
    for i in range(len(train_masks)):
        if(np.max(train_masks[i])<7):
            train_masks_cleaned.append(train_masks[i])
            train_images_cleaned.append(train_images[i])
            train_images_cleaned2.append(train_images2[i])


    train_masks=np.asarray(train_masks_cleaned)
    train_images=np.asarray(train_images_cleaned)
    train_images2=np.asarray(train_images_cleaned2)


    #
    # for i in range(len(train_masks)):
    #     plt.imshow(train_masks[i])
    #     plt.show()
    #
    #     plt.imshow(train_images[i][:,:,2])
    #     plt.show()

    print (train_images.shape)






    #need additional dim for tensor format
    # train_images=np.expand_dims(train_images,axis=3)

    train_images=normalize(train_images,axis=1)

    train_masks=np.expand_dims(train_masks,axis=3)

    print(np.unique(train_masks))

    masks_copy=train_masks.copy()
    masks_copy=np.expand_dims(masks_copy, axis=3)


    train_masks=to_categorical(train_masks, num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(train_images,train_masks, test_size=0.1, random_state=2 )

    print (y_train.shape)

    cw=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(np.ravel(masks_copy, order='C')), y=np.ravel(masks_copy, order='C'))


    # #to dictionary
    # cw = {i : cw[i] for i in range(5)}


    print("class weights: ",cw)


    ##test
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy


        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss
    ##endtst

    dice_loss=sm.losses.DiceLoss(class_weights=cw)
    focal_loss=sm.losses.CategoricalFocalLoss()
    # total_loss= dice_loss + focal_loss
    #total_loss=dice_loss

    catce=sm.losses.CategoricalCELoss()

    total_loss=(1/2)*focal_loss+catce
    # total_loss=sm.losses.JaccardLoss()
    # total_loss=

    #If smooth is set too low,
    # #when the ground truth has few to 0 white pixels and the predicted image
    # has some non-zero number of white pixels, the model will be penalized more heavily.


    def dice_coef(y_true, y_pred, smooth=10):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice


    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)



    patch_size=X_train.shape[1]
    im_channels=X_train.shape[3]



    def get_model():
        return multi_model(n_classes=n_classes, patch_size=patch_size, num_bands=im_channels)


    model = get_model()
    # model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy'])



    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics)
    #

    #this
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=metrics)


    learn_r= trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
    opti= trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])


    print(opti)

    if(str(opti)=="Adam"):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_r), loss=total_loss, metrics=metrics)
    if (str(opti) == "RMSprop"):
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_r), loss=total_loss, metrics=metrics)
    if (str(opti) == "SGD"):
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learn_r), loss=total_loss, metrics=metrics)

    # model.compile(optimizer='adam', loss=total_loss, metrics=metrics)


    model.summary()


    #write custom loss fn that's less harsh



    m1=model.fit(X_train, y_train, batch_size=trial.suggest_int('bsize', 2,128),verbose=1, epochs=150, validation_data=(X_test,y_test), shuffle=True)
    #m1=model.fit(X_train, y_train, batch_size=8,verbose=1, epochs=80, validation_data=(X_test,y_test), shuffle=True) # , class_weight=cw)

    #try romoving class weights when using dice lsos since it already considers them

    #try combining crossentropy and focal loss



    weightsave='test1_weights_trial%s.hdf5'%(trial.number)

    model.save(weightsave)

    model.load_weights(weightsave)


    _, acc, *rest2= model.evaluate(X_test, y_test)
    print("accuracy: ", (acc*100.0), "%")




    loss= m1.history['loss']
    val_loss=m1.history['val_loss']
    epochs=range(1,len(loss)+1)
    plt.plot(epochs, loss, 'y', label='training loss')
    plt.plot(epochs, val_loss, 'r', label='validation loss')
    plt.title('training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_trial%s.png'%(trial.number))
    plt.cla()
    plt.clf()
    plt.close("all")

    return acc



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=350)




