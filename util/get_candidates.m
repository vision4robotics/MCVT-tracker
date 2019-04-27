function object_parts_box = get_candidates(im_c, sz, scale)
if isscalar(sz),  %square sub-window
	sz = [sz, sz];
end
    
half_h = sz(1)/4;
half_w = sz(2)/4;
    
im_h = size(im_c, 1);
im_w = size(im_c, 2);
    
w_num = ceil(im_w/half_w);
h_num = ceil(im_h/half_h);
    
% object_parts_pos = cell(w_num-1, h_num-1);
object_parts_box = zeros((w_num-1)*(h_num-1), 4);
c = 0;
for i = 1:w_num
    for j = 1:h_num
        if i < w_num && j < h_num
             tmp_pos = [j * half_h, i * half_w];
             for s = 1:numel(scale)
                tmp_box = [tmp_pos([2,1]) - scale(s).*sz([2,1])/2, scale(s).*sz([2,1])];
                c = c + 1;
                object_parts_box(c, :) = tmp_box;
             end
        end
    end
end

object_parts_box(:,1) = max(1-object_parts_box(:,3)/2,min(im_w-object_parts_box(:,3)/2, object_parts_box(:,1)));
object_parts_box(:,2) = max(1-object_parts_box(:,4)/2,min(im_h-object_parts_box(:,4)/2, object_parts_box(:,2)));
object_parts_box(:,1) = max(1, object_parts_box(:,1));
object_parts_box(:,2) = max(1, object_parts_box(:,2));
object_parts_box = ceil(object_parts_box);

end