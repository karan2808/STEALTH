# STEALTH Stereo Analysis Using Low Texture Thermal Images.

Brief Introduction:

We tackle the problem of computing disparity or depth of objects in a scene using Infrared Images. As far as we know, this is the first repository to deal with disparity computation using Infrared Images. 
We have selected GANet and AANet for this task as the baseline models.

Usage:

AANet: To train the model from the beginning, run the following command,

python train.py --data_dir $Path to Dataset$ --dataset_name $Dataset Name$ --mode train --checkpoint_dir $Checkpoint Directory$ --batch_size 2 --val_batch_size       1 --img_height 480 --img_width 636 --val_img_height 480 --val_img_width 636 --feature_type aanet --feature_pyramid_network --highest_loss_only --learning_rate 1e-4 --milestones 400,600,800,900 --max_epoch 1000 --save_ckpt_freq 100 --no_validate --max_disp 96 --no_intermediate_supervision.

To test your model, run the following prediction script,

python predict.py --data_dir demo --pretrained_aanet $model path$ --feature_type aanet --feature_pyramid_network --no_intermediate_supervision

GANet:To train the model from the beggining, run the following command,

python train.py --batchSize=2 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=96 \
                --thread=1 \
                --data_path=$Path to the dataset$ \
                --training_list=$Path to the list with image names$ \
                --save_path='./checkpoint/cmuset_ckpt' \
                --kitti2015=1 \
                --shift=3 \
                --nEpochs=1000
                
To test your model, run the following prediction script.

python predict.py --crop_height=512 \
                  --crop_width=640 \
                  --max_disp=96 \
                  --data_path='$Path to the dataset.' \
                  --test_list='lists/CMU_set_val_list.list' \
                  --save_path='./result_cmu_set/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/cmu_set_ckpt.pth'
                  
Acknowledgements: Part of our code is adopted from previous works, AANet (https://github.com/haofeixu/aanet) and GANet (https://github.com/feihuzhang/GANet), which deal with RGB Images. We have used the CATS (A Color And Thermal Stereo Dataset, http://bigdatavision.org/CATS/) for training and evaluation.
