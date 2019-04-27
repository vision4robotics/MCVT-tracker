function [out_npca, out_pca] = get_subwindow_cn(im, pos, model_sz, non_pca_features, pca_features, w2c, currentScaleFactor)

if isscalar(model_sz)  %square sub-window
    model_sz = [model_sz, model_sz];
end

patch_sz = floor(model_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end
if patch_sz(2) < 1
    patch_sz(2) = 2;
end

xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
im_patch = im(ys, xs, :);

%resize image to model size
im_patch = mexResize(im_patch, [floor(model_sz(1)/4),floor(model_sz(2)/4)], 'auto');

% compute non-pca feature map
if ~isempty(non_pca_features)
    out_npca = get_feature_map_cn(im_patch, non_pca_features, w2c);
else
    out_npca = [];
end

% compute pca feature map
if ~isempty(pca_features)
    temp_pca = get_feature_map_cn(im_patch, pca_features, w2c);
    out_pca = reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);
else
    out_pca = [];
end
end

