import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

#from CyclicLearningRate import CyclicLR
#from datasets import build_batch_generator, generate_filenames
from losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border
from models import make_model
from params import args
from utils import freeze_model, ThreadsafeIter
from keras.preprocessing.image import array_to_img

prediction_dir = args.pred_mask_dir

output_dir = args.pred_mask_dir

test_data_dir = os.path.join(args.dataset_dir,args.test_data_dir)

print(test_data_dir)

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
idata_gen_args = dict(horizontal_flip=False,
                     vertical_flip=False,
                     zoom_range=0,
                     rescale=1./255,
                     preprocessing_function=preprocess_input)
mdata_gen_args = dict(horizontal_flip=False,
                     vertical_flip=False,
                     zoom_range=0,
                     rescale=1./255,
                     )
image_datagen = ImageDataGenerator(**idata_gen_args)
mask_datagen = ImageDataGenerator(**mdata_gen_args)

seed = 1
bs=1

image_generator = image_datagen.flow_from_directory(
    test_data_dir,
    seed=seed,
    batch_size=bs,
    color_mode='rgb',
    shuffle=False)

model = make_model((None, None, 3))
model.load_weights(args.weights)

import matplotlib.pyplot as plt
def predict_one(filename):
    image_batch, y = next(image_generator)
    predicted_mask_batch = model.predict(image_batch)
    image = image_batch[0]
    #print(image.shape)
    predicted_mask = predicted_mask_batch[0].reshape((256,256,1))
    #print(predicted_mask.shape)
    #print(predicted_mask)
    #plt.imshow(image[:,:,0], cmap='gray')
    #plt.imshow(predicted_mask[:,:,0], alpha=0.6)
        #filename = filenames[i * batch_size + j]
    #prediction = preds[j][:, 1:-1, :]
    #print(predicted_mask[:,:,0].min())
    #print(predicted_mask[:,:,0].max())
    array_to_img(predicted_mask*255).save(os.path.join(args.dataset_dir,output_dir, filename+'.jpg'))
    #array_to_img(image*255).save(os.path.join(args.dataset_dir,output_dir, filename+'image.jpg'))

for i in range(1,len(image_generator)):
    predict_one(str(i))
