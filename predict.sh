python predictLoopWSM.py \
  --weights=models/inception_resnet_v2--loss-bce_dice-fold_None-2560.001000-30-0.1071685-0.9206451-92.2793160.h5\
  --dataset_dir='/data/user/wsmonroe/MetalsGroup/pennyProject/data'\
  --test_data_dir='images/chopped'\
  --pred_mask_dir='predicted_masks'\
  --network='resnet50'\
  --preprocessing_function='tf'\
  --predict_on_val=False\
  --input_height=256\
  --input_width=256

#CUDA_VISIBLE_DEVICES='0,1' python predict_multithreaded.py \
#  --gpu=0,1\
#  --weights=weights/mobilenet-loss-bce_dice-fold_0-10240.000010-1-0.0034889-0.9967070-99.7011537.h5\
#  --test_data_dir=input/train\
#  --pred_mask_dir=predicted_masks/mobilenet_val_from_fold_0\
#  --fold=0\
#  --folds_source=folds.csv\
#  --dataset_dir=input\
#  --network='mobilenet'\
#  --preprocessing_function='tf'\
#  --predict_on_val=True
#python train.py\
#  --seed=80\
#  --dataset_dir='/data/user/wsmonroe/MetalsGroup/pennyProject/data'\
#  --network='mobilenet'\
#  --preprocessing_function='tf'\
#  --learning_rate=0.001\
#  --loss_function='bce_dice'\
#  --train_data_dir_name='images/chopped'\
#  --val_data_dir_name='images/chopped'\
#  --val_mask_dir='masks/labels'\
#  --train_mask_dir='masks/labels'\
#  --input_height=256\
#  --input_width=256
