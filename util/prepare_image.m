function input_image  = prepare_image(img, imageSz, pixel_means)
% prepare image for caffe networks

if size(img, 3) == 1
    img = cat(3, img, img, img);
end

input_image = single(img);
input_image = imresize(input_image, [imageSz imageSz], 'bilinear');
input_image = input_image(:, :, [3 2 1]);
input_image = bsxfun(@minus, input_image, pixel_means);
input_image = permute(input_image, [2 1 3]);

input_image = single(input_image);
end