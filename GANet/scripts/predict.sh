
python predict.py --crop_height=512 \
                  --crop_width=640 \
                  --max_disp=96 \
                  --data_path='/home/rtml/aditi/CMU_Set/CMU_SET/' \
                  --test_list='lists/CMU_set_val_list.list' \
                  --save_path='./result_cmu_set/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/cmu_set_ckpt.pth'
exit

# python predict.py --crop_height=384 \
#                   --crop_width=1248 \
#                   --max_disp=192 \
#                   --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
#                   --test_list='lists/kitti2012_test.list' \
#                   --save_path='./result/' \
#                   --kitti=1 \
#                   --resume='./checkpoint/kitti2012_final.pth'



