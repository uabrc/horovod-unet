import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from CyclicLearningRate import CyclicLR
#from datasets import build_batch_generator, generate_filenames
from losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border
from models import make_model
from params import args
from utils import freeze_model, ThreadsafeIter

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    mask_dir = os.path.join(args.dataset_dir, args.train_mask_dir_name)
    val_mask_dir = os.path.join(args.dataset_dir, args.val_mask_dir_name)

    train_data_dir = os.path.join(args.dataset_dir, args.train_data_dir_name)
    val_data_dir = os.path.join(args.dataset_dir, args.val_data_dir_name)

    if args.net_alias is not None:
        formatted_net_alias = '-{}-'.format(args.net_alias)

    best_model_file =\
        '{}/{}{}loss-{}-fold_{}-{}{:.6f}'.format(args.models_dir, args.network, formatted_net_alias, args.loss_function, args.fold, args.input_width, args.learning_rate) +\
        '-{epoch:d}-{val_loss:0.7f}-{val_dice_coef:0.7f}-{val_dice_coef_clipped:0.7f}.h5'

    model = make_model((None, None, args.stacked_channels + 3))
    freeze_model(model, args.freeze_till_layer)

    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()

    model.compile(loss=make_loss(args.loss_function),
                  optimizer=optimizer,
                  metrics=[dice_coef_border, dice_coef, binary_crossentropy, dice_coef_clipped])

    if args.show_summary:
        model.summary()

    crop_size = None

    if args.use_crop:
        crop_size = (args.input_height, args.input_width)
        print('Using crops of shape ({}, {})'.format(args.input_height, args.input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')

    #folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    #train_ids = generate_filenames(folds_df[folds_df.fold != args.fold]['id'])
    #val_ids = generate_filenames(folds_df[folds_df.fold == args.fold]['id'])
    #print('Training fold #{}, {} in train_ids, {} in val_ids'.format(args.fold, len(train_ids), len(val_ids)))


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

    seed = 1
    bs=6


    image_generator = image_datagen.flow_from_directory(
        '/data/user/wsmonroe/MetalsGroup/pennyProject/pennyData/TrainingSet/preprocessed/data',
        seed=seed,
        batch_size=bs,
        color_mode='rgb',
        target_size=(512,512))

    mask_generator = mask_datagen.flow_from_directory(
        '/data/user/wsmonroe/MetalsGroup/pennyProject/pennyData/TrainingSet/preprocessed/labels',
        seed=seed,
        batch_size=bs,
        color_mode='grayscale',
        target_size=(512,512))

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
    vimage_datagen = ImageDataGenerator(**vdata_gen_args)
    vmask_datagen = ImageDataGenerator(**vmdata_gen_args)
    vimage_generator = vimage_datagen.flow_from_directory(
        '/data/user/wsmonroe/MetalsGroup/pennyProject/pennyData/TrainingSet/preprocessed/validation/data',
        seed=seed,
        batch_size=bs,
        color_mode='rgb',
        target_size=(512,512))

    vmask_generator = vmask_datagen.flow_from_directory(
        '/data/user/wsmonroe/MetalsGroup/pennyProject/pennyData/TrainingSet/preprocessed/validation/labels',
        seed=seed,
        batch_size=bs,
        color_mode='grayscale',
        target_size=(512,512))
    val_generator = combine_generator(vimage_generator,vmask_generator)#zip(image_generator, mask_generator)
    '''train_generator = build_batch_generator(
        train_ids,
        img_dir=train_data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        mask_dir=mask_dir,
        aug=True
    )

    val_generator = build_batch_generator(
        val_ids,
        img_dir=val_data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        out_size=(args.out_height, args.out_width),
        crop_size=None,
        mask_dir=val_mask_dir,
        aug=False
    )'''

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=False,
                                                  save_weights_only=True)

    callbacks = [best_model, EarlyStopping(patience=45, verbose=10)]
    if args.clr is not None:
        clr_params = args.clr.split(',')
        base_lr = float(clr_params[0])
        max_lr = float(clr_params[1])
        step = int(clr_params[2])
        mode = clr_params[3]
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step, mode=mode)
        callbacks.append(clr)
    model.fit_generator(
        train_generator,
        steps_per_epoch=3594 / bs + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=3594 / bs + 1,
        callbacks=callbacks,
        max_queue_size=50,
        workers=8)

if __name__ == '__main__':
    main()
