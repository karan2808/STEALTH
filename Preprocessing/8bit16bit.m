clc;
clear all;
close all;

a = imread('CARS_disp/1.png');
%a = importdata('CARS_disp/interp_1.txt');
%a = uint8(a);
a = uint16((double(a)/255)*65535);
imwrite(a,'CARS_disp/1_16.png');