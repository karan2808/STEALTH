img1 = load('data/gt_disparity.txt');
% img1 = round((img1)*65535);
% img1 = mat2gray(img1);
% imshow(img1)
% imwrite(img1, 'data/gt_disp_mat.png')
colormap turbo
imagesc(img1)
% img1 = imread('data/computed_disp_sgmM.png');
% img1 = rgb2gray(img1);
% img1 = double(img1);
% img1 = round((img1/255)*65535);
% imwrite(img1, 'data/computed_disp.png')