#CUDA_VISIBLE_DEVICES='0,1' python predict_multithreaded.py \
 # --gpu=0,1\
 # --weights=weights/mobilenet-loss-bce_dice-fold_3-10240.000010-23-0.0035732-0.9965180-99.6974207.h5\
 # --test_data_dir=input/train\
 # --pred_mask_dir=predicted_masks/mobilenet_val_from_fold_3\
 # --fold=3\
 # --folds_source=folds.csv\
 # --dataset_dir=input\
 # --network='mobilenet'\
 # --preprocessing_function='tf'\
 # --predict_on_val=True
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
python train.py\
  --seed=80\
  --dataset_dir='/data/user/wsmonroe/MetalsGroup/pennyProject/pennyData/TrainingSet'\
  --network='inception_resnet_v2'\
  --preprocessing_function='tf'\
  --learning_rate=0.001\
  --loss_function='bce_dice'\
  --train_data_dir_name='preprocessed/data'\
  --val_data_dir_name='preprocessed/validation/data'\
  --val_mask_dir='preprocessed/validation/data/labels'\
  --train_mask_dir='preprocessed/labels'\
  --input_height=256\
  --input_width=256\
  --epochs=100
