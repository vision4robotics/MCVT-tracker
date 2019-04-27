function [S, new_rect]= get_surrounding(last_rect, I, lamda)
% get the surrounding region for object searching
center      = [last_rect(2)+last_rect(4)/2 last_rect(1)+last_rect(3)/2]; 
width       = last_rect(3);
height      = last_rect(4);
new_width   = lamda * sqrt(width.^2 + height.^2);
new_height  = new_width;
new_rect    = [center(2)-new_width/2 center(1)-new_height/2 new_width, new_height];
new_rect    = floor(new_rect);
new_rect(1) = max(new_rect(1), 1);
new_rect(2) = max(new_rect(2), 1);
S = imcrop(I, new_rect);
end