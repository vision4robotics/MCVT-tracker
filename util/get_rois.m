function input_rois = get_rois(box, imageSz, img)
% get roi of object
box_num   = size(box, 1);    % number of boxes
input_rois = zeros(box_num, 5);

for i = 1:box_num
    input_rois(i, 2) = box(i, 1) * imageSz / size(img, 2);
    input_rois(i, 3) = box(i, 2) * imageSz / size(img, 1);
    input_rois(i, 4) = (box(i, 1) + box(i, 3) - 1) * imageSz / size(img, 2);
    input_rois(i, 5) = (box(i, 2) + box(i, 4) - 1) * imageSz / size(img, 1);
    input_rois(i, 2:end) = input_rois(i, 2:end) - 1;
end

input_rois = single(input_rois);
end