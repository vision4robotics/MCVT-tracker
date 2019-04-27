function pos = rectify_tracker(im, pos, t_sz, evaluate_param)
global net;

im_color = im;
if size(im_color, 3) == 1
    im_color = cat(3, im_color, im_color, im_color); 
end

lamda         = evaluate_param.lamda;
rect          = [pos([2,1]) - t_sz([2,1])/2, t_sz([2,1])];

% get the surrounding region of current position of the target object
[new_im_color, new_rect] = get_surrounding(rect, im_color, lamda);    

% get probable regions (candidates) from the surrounding region (sliding window strategy)
object_cand_boxes = get_candidates(new_im_color, t_sz, evaluate_param.scale);

object_cand_boxes(:, 1) = object_cand_boxes(:, 1) + new_rect(1);
object_cand_boxes(:, 2) = object_cand_boxes(:, 2) + new_rect(2);

if isempty(object_cand_boxes)
    object_cand_boxes=rect;
end
% obtain features for all candidates
input_roi   = get_rois(object_cand_boxes, evaluate_param.imageSz, im_color);
input_im    = prepare_image(im_color, evaluate_param.imageSz, evaluate_param.pixel_means);

input_blobs    = cell(2, 1);
input_blobs{1} = input_im;
net.blobs('rois').reshape([5, size(input_roi, 1)]);
input_blobs{2} = input_roi';

blobs_out    = net.forward(input_blobs);

out          = blobs_out{1};
tfeat        = squeeze(out(size(out, 1), size(out, 2), :, :));

% compute rectification score for each candidate within one batch
tmp_score    = tfeat' * evaluate_param.firstframe_feat;
[~, max_ids]  = sort(tmp_score, 'descend');

% get the region with the highest score
max_id       = max_ids(1);
m_score      = tmp_score(max_id);
tmp_box      = object_cand_boxes(max_id, :);
m_pos        = [tmp_box(2)+tmp_box(4)/2 tmp_box(1)+tmp_box(3)/2];

% score = m_score;
if m_score >= evaluate_param.det_threshold
    pos = m_pos;
end

end