function evaluate_param = init_evaluation(img_files, init_box)
% parameters for second-stage evaluation

evaluate_param.imageSz         = 512;
evaluate_param.init_box        = init_box;
evaluate_param.scale           = [1];
evaluate_param.firstframe      = imread(img_files{1});
evaluate_param.lamda = 1.6;

evaluate_param.threshold_all = 1.1;
evaluate_param.det_threshold = 1.3;

if size(evaluate_param.firstframe, 3) == 1
    evaluate_param.firstframe = cat(3, evaluate_param.firstframe, evaluate_param.firstframe, evaluate_param.firstframe);
end

% Pixel means of all three RGB channels 
evaluate_param.pixel_means = reshape([104.007 116.669 122.679], [1 1 3]);
end