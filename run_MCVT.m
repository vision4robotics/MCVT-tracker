function results=run_MCVT(seq)

addpath('util\');
addpath('E:\Final\Program\caffe-windows-ms\matlab');   % caffe\matlab


% deploy document and caffe model for the second-stage Siamese networks
def       = 'siamese_networks\deploy.prototxt';        
weight    = 'siamese_networks\similarity.caffemodel';  

% load networks
global net;
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(def, weight, 'test');

% load parameters for fDSST tracker
fDSST_param;

    ground_truth = seq.ground_truth;
    img_files = seq.s_frames;
    init_box = seq.init_rect; 
    video_path = strcat(seq.video_path,'\');
    target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);

    
    % parameters for second-stage 
    verify_param = init_evaluation(img_files, init_box);
    
    % prepare image for caffe networks
    input_im     = prepare_image(verify_param.firstframe, verify_param.imageSz, verify_param.pixel_means);
    input_roi    = get_rois(verify_param.init_box, verify_param.imageSz, verify_param.firstframe);

    % extract feature for object in the first frame
    firstframe_input_blobs    = cell(2, 1);
    firstframe_input_blobs{1} = input_im;
    net.blobs('rois').reshape([5, size(input_roi, 1)]);
    firstframe_input_blobs{2} = input_roi';

    blobs_out                    = net.forward(firstframe_input_blobs);
    verify_param.firstframe_feat = squeeze(blobs_out{1});      % feature for the target

    params.init_pos = floor(pos);
    params.wsize = floor(target_sz);
    params.s_frames = img_files;
    params.video_path = video_path;

    % do tracking
    results = MCVT(params, verify_param);

% delete caffe model from memory
caffe.reset_all();
end