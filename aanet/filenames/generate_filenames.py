import os
from glob import glob


def gen_kitti_2015():
    data_dir = 'data/KITTI/kitti_2015/data_scene_flow'

    train_file = 'KITTI_2015_train.txt'
    val_file = 'KITTI_2015_val.txt'

    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        dir_name = 'image_2'
        left_dir = os.path.join(data_dir, 'training', dir_name)
        left_imgs = sorted(glob(left_dir + '/*_10.png'))

        print('Number of images: %d' % len(left_imgs))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'image_3')
            disp_path = left_img.replace(dir_name, 'disp_occ_0')

            img_id = int(os.path.basename(left_img).split('_')[0])

            if img_id % 5 == 0:
                val_f.write(left_img.replace(data_dir + '/', '') + ' ')
                val_f.write(right_img.replace(data_dir + '/', '') + ' ')
                val_f.write(disp_path.replace(data_dir + '/', '') + '\n')
            else:
                train_f.write(left_img.replace(data_dir + '/', '') + ' ')
                train_f.write(right_img.replace(data_dir + '/', '') + ' ')
                train_f.write(disp_path.replace(data_dir + '/', '') + '\n')


def gen_cats_ir():

    data_dir = '/home/rtml/Siddhant/CATS_INFERNO/CATS_INFERNO'
    #data_dir = '/home/rtml/Siddhant/CATS_NEW_OUTDOOR'
    
    #data_dir = '/home/rtml/Siddhant/CATS_NEW_GT'

    #data_dir = '/home/rtml/Siddhant/CATS_CLIP_GT'
    
    #data_dir = '/home/rtml/Siddhant/CATS_3C_FULL'

    #train_file = 'CATS_New_train.txt'
    #val_file = 'CATS_New_val.txt'

    #train_file = 'CATS_outdoor_train.txt'
    #val_file = 'CATS_outdoor_val.txt'

    #train_file = 'CATS_Clip_train.txt'
    #val_file = 'CATS_Clip_val.txt'

    #train_file = 'CATS_All_train.txt'
    #val_file = 'CATS_All_val.txt'

    train_file = 'CATS_Inferno_train.txt'
    val_file = 'CATS_Inferno_val.txt'

    

    # Split data into train and validation
    with open(train_file, 'w') as train_f, open(val_file,'w') as val_f:
        dir_name = 'left'
        #left_dir = os.path.join(data_dir, 'All', dir_name)
        left_dir = os.path.join(data_dir,dir_name)
        left_imgs = (glob(left_dir + '/*.png'))

        print('Number of images: %d' % len(left_imgs))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'right')
            disp_path = left_img.replace(dir_name, 'disp')

            img_id = int(os.path.basename(left_img).split('.')[0])

            #if img_id % 10 == 0:
            if img_id % 8 == 0:
                val_f.write(left_img.replace(data_dir + '/', '') + ' ')
                val_f.write(right_img.replace(data_dir + '/', '') + ' ')
                val_f.write(disp_path.replace(data_dir + '/', '') + '\n')
            else:
                train_f.write(left_img.replace(data_dir + '/', '') + ' ')
                train_f.write(right_img.replace(data_dir + '/', '') + ' ')
                train_f.write(disp_path.replace(data_dir + '/', '') + '\n')



if __name__ == '__main__':
    #gen_kitti_2015()
    gen_cats_ir()
