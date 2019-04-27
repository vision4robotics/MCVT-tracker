function [ cell_gray ] = cell_grayscale( img, w )

if size(img,3) == 3
   %convert to grayscale
   gray_image = rgb2gray(img);
else
   gray_image = img;
end
gray_image = single(gray_image);

%compute the integral image
iImage = integralImage(gray_image);
i1 = (w:w:size(gray_image,1)) + 1;
i2 = (w:w:size(gray_image,2)) + 1;
cell_sum = iImage(i1,i2) - iImage(i1,i2-w) - iImage(i1-w,i2) + iImage(i1-w,i2-w);
cell_gray = cell_sum / (w*w * 255) - 0.5;
end