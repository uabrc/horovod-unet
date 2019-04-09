import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border
from models import make_model
from params import args
from utils import freeze_model, ThreadsafeIter

import argparse
import keras
from keras import backend as K
import horovod.keras as hvd
import tensorflow as tf
import os

#
# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


mask_dir = os.path.join(args.dataset_dir, args.train_mask_dir_name)
val_mask_dir = os.path.join(args.dataset_dir, args.val_mask_dir_name)

train_data_dir = os.path.join(args.dataset_dir, args.train_data_dir_name)
val_data_dir = os.path.join(args.dataset_dir, args.val_data_dir_name)

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break


# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

##Create data generators for training and validation datasets
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.2,
                     rescale=1./255,
                     preprocessing_function=preprocess_input
                    )
mdata_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                    rescale=1./255,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**mdata_gen_args)

seed = args.seed

bs=args.batch_size

image_generator = image_datagen.flow_from_directory(
    train_data_dir,
    seed=seed,
    batch_size=bs,
    color_mode='rgb',
    target_size=(args.out_height, args.out_width))

mask_generator = mask_datagen.flow_from_directory(
    mask_dir,
    seed=seed,
    batch_size=bs,
    color_mode='grayscale',
    target_size=(args.out_height, args.out_width))

def combine_generator(gen1,gen2):
    while True:

        X = gen1.next()
        y = gen2.next()

        yield(X[0],y[0])

train_generator = combine_generator(image_generator,mask_generator)#zip(image_generator, mask_generator)

vdata_gen_args = dict(
                     rescale=1./255,
                     preprocessing_function=preprocess_input
                    )
vmdata_gen_args = dict(
                    rescale=1./255,
                     )
bsv=2
vimage_datagen = ImageDataGenerator(**vdata_gen_args)
vmask_datagen = ImageDataGenerator(**vmdata_gen_args)
vimage_generator = vimage_datagen.flow_from_directory(
    val_data_dir,
    seed=seed,
    batch_size=bsv,
    color_mode='rgb',
    target_size=(args.out_height, args.out_width))

vmask_generator = vmask_datagen.flow_from_directory(
    val_mask_dir,
    seed=seed,
    batch_size=bsv,
    color_mode='grayscale',
    target_size=(args.out_height, args.out_width))
val_generator = combine_generator(vimage_generator,vmask_generator)#zip(image_generator, mask_generator)


#model generation###############################
model = make_model((None, None, args.stacked_channels + 3))
freeze_model(model, args.freeze_till_layer)

if args.weights is None:
    print('No weights passed, training from scratch')
else:
    print('Loading weights from {}'.format(args.weights))
    model.load_weights(args.weights, by_name=True)



#if args.show_summary:
 #   model.summary()



if hvd.rank() == 0:
    model.summary()

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
#resume_from_epoch=0
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                           compression=compression)
else:
    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    #model = keras.models.Model.from_config(model_config)

    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                               momentum=args.momentum)
    #optimizer = Adam(lr=args.learning_rate*hvd.size())
                        #momentum=args.momentum)
    # Horovod: add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)

    model.compile(loss=make_loss(args.loss_function),
              optimizer=optimizer,
              metrics=[dice_coef_border, dice_coef, binary_crossentropy, dice_coef_clipped])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    #hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, end_epoch=100, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=100, end_epoch=130, multiplier=1e-2),
    #hvd.callbacks.LearningRateScheduleCallback(start_epoch=160, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=130, end_epoch=160, multiplier=1e-3),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=160, end_epoch=190, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=190, multiplier=1e-3),
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data helps to increase probability that every validation
# example will be evaluated.
model.fit_generator(train_generator,
                    steps_per_epoch=3954 // hvd.size(),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=6,
                    initial_epoch=resume_from_epoch,
                    validation_data=val_generator,
                    validation_steps=(264/bsv) // hvd.size())


# Evaluate the model on the full data set.
score = hvd.allreduce(model.evaluate_generator(val_generator, len(val_generator), workers=6))
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
