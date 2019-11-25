% =========================================================================

% =========================================================================

close all;
clear all;

%% read ground truth image
% im  = imread('Set5\baby_GT.bmp');
%im  = imread('Set14\zebra.bmp');
im = imread('Set14/monarch.bmp');
%% set parameters
up_scale = 3;
model='SRCNN\x3.mat'
model = 'model\9-5-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-3-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-1-5(91 images)\x3.mat';
% up_scale = 2;
% model = 'model\9-5-5(ImageNet)\x2.mat'; 
% up_scale = 4;
% model = 'model\9-5-5(ImageNet)\x4.mat';

%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');

%% SRCNN
im_h = SRCNN(model, im_b);
%% Multi-channel SRCNN
im_m = MP_SRCNN(model, im_b);

%% parallel SRCNN 
im_p = P_SRCNN(model, im_b);

%% 4ch-R
im_r = CH4_R_SRCNN(model, im_b);

%% remove border
im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);

im_m = shave(uint8(im_m * 255), [up_scale, up_scale]);
im_p = shave(uint8(im_p * 255), [up_scale, up_scale]);
im_r = shave(uint8(im_r * 255), [up_scale, up_scale]);


%% compute PSNR
psnr_bic = compute_psnr(im_gnd,im_b);
psnr_srcnn = compute_psnr(im_gnd,im_h);
psnr_MP_srcnn = compute_psnr(im_gnd,im_m);

psnr_P_srcnn = compute_psnr(im_gnd,im_p);
psnr_R_srcnn = compute_psnr(im_gnd,im_r);

%% show results
fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
fprintf('PSNR for Multi channel SRCNN Reconstruction: %f dB\n', psnr_MP_srcnn);

fprintf('PSNR for Parallel SRCNN Reconstruction: %f dB\n', psnr_P_srcnn);
fprintf('PSNR for 4 channel rotary SRCNN Reconstruction: %f dB\n', psnr_R_srcnn);

subplot(2,1,1);
bigimg = [im_gnd im_b im_h];
imshow(bigimg); title('src                      bicubic                         srcnn')


subplot(2,1,2);
bigimg = [im_m im_p im_r];
imshow(bigimg); title('multi channels srcnn         parallel srcnn             multi channel rotary srcnn ')


% figure, imshow(im_b); title('Bicubic Interpolation');
% figure, imshow(im_h); title('SRCNN Reconstruction');

%imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
%imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);
