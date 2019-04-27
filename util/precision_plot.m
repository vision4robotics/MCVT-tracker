function precisions = precision_plot(positions, ground_truth)
max_threshold = 50;  %used for graphs in the paper
precisions = zeros(max_threshold, 1);
	
if size(positions,1) ~= size(ground_truth,1),
	n = min(size(positions,1), size(ground_truth,1));
	positions(n+1:end,:) = [];
	ground_truth(n+1:end,:) = [];
end

%calculate distances to ground truth over all frames
distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + (positions(:,2) - ground_truth(:,2)).^2);
distances(isnan(distances)) = [];

%compute precisions
for p = 1:max_threshold
    precisions(p) = nnz(distances <= p) / numel(distances);
end
end

