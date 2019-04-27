function results = MCVT(params, evaluate_param)
global net;
% global sim_score;

s_frames      = params.s_frames;
pos_all       = floor(params.init_pos);
target_sz     = floor(params.wsize * params.resize_factor);

visualization = params.visualization;

cn_non_compressed_features = params.cn_non_compressed_features;
cn_compressed_features = params.cn_compressed_features;
cn_num_compressed_dim = params.cn_num_compressed_dim;

% load the normalized Color Name matrix
temp = load('w2crs');
w2c = temp.w2crs;

num_frames = numel(s_frames); 

init_target_sz = target_sz;

if prod(init_target_sz) > params.translation_model_max_area
    currentScaleFactor = sqrt(prod(init_target_sz) / params.translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

%window size, taking padding into account
sz = floor( base_target_sz * (1 + params.padding ));

featureRatio = 4;

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * params.output_sigma_factor;
use_sz = floor(sz/featureRatio);
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);

[rs, cs] = ndgrid( rg,cg); 
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

interp_sz = size(y) * featureRatio;

cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );

if params.number_of_scales > 0
    scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor;
    
    scale_exp = (-floor((params.number_of_scales-1)/2):ceil((params.number_of_scales-1)/2)) * params.number_of_interp_scales/params.number_of_scales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((params.number_of_scales-1)/2)]);
    
    interp_scale_exp = -floor((params.number_of_interp_scales-1)/2):ceil((params.number_of_interp_scales-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((params.number_of_interp_scales-1)/2)]);
    
    scaleSizeFactors = params.scale_step .^ scale_exp;
    interpScaleFactors = params.scale_step .^ interp_scale_exp_shift;
    
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
    ysf = single(fft(ys));
    scale_window = single(hann(size(ysf,2)))'; 
    
    %make sure the scale model is not to large, to save computation time
    if params.scale_model_factor^2 * prod(init_target_sz) > params.scale_model_max_area
        params.scale_model_factor = sqrt(params.scale_model_max_area/prod(init_target_sz));
    end
    
    %set the scale model size
    scale_model_sz = floor(init_target_sz * params.scale_model_factor);
    
    im = imread(s_frames{1});
    
    %force reasonable scale changes
    min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ sz)) / log(params.scale_step));
    max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));
    
    max_scale_dim = strcmp(params.s_num_compressed_dim,'MAX');
    if max_scale_dim
        s_num_compressed_dim = length(scaleSizeFactors);
    else
        s_num_compressed_dim = params.s_num_compressed_dim;
    end
end

% initialize the projection matrix
projection_matrix = [];
projection_matrix_cn = [];

rect_position_all = zeros(num_frames, 4);
second_stage_flag=0;
time = 0;
for frame = 1:num_frames
    %load image
    im = imread(s_frames{frame});
    tic();
    verification_result = true;
    %do tracking
    if frame > 1
        iter = 1;
       
        %translation search
        while iter <= params.refinement_iterations
            [xt_npca, xt_pca] = get_subwindow(im, pos_all, sz, currentScaleFactor);
            [xt_npca_gray, xt_pca_gray] = get_subwindow_gray(im, pos_all, sz, currentScaleFactor);
            [xt_npca_cn, xt_pca_cn] = get_subwindow_cn(im, pos_all, sz, cn_non_compressed_features, cn_compressed_features, w2c, currentScaleFactor);
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window);
            xtf = fft2(xt);
            xt_gray = feature_projection(xt_npca_gray, xt_pca_gray, 1, cos_window);
            xtf_gray = fft2(xt_gray);
            xt_cn = feature_projection_cn(xt_npca_cn, xt_pca_cn, projection_matrix_cn, cos_window);
            xtf_cn = fft2(xt_cn);                    
            
            responsef = sum(hf_num .* xtf, 3) ./ (hf_den + params.lambda);
            responsef_gray  = sum(hf_num_gray  .* xtf_gray , 3) ./ (hf_den_gray  + params.lambda);
            responsef_cn = sum(hf_num_cn .* xtf_cn, 3) ./ (hf_den_cn + params.lambda);
            
            % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if params.interpolate_response > 0
                if params.interpolate_response == 2
                    % use dynamic interp size
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end
                responsef = resizeDFT2(responsef, interp_sz); 
                responsef_gray = resizeDFT2(responsef_gray, interp_sz);
                responsef_cn = resizeDFT2(responsef_cn, interp_sz);
            end
            
            response = ifft2(responsef, 'symmetric');
            response_gray = ifft2(responsef_gray, 'symmetric');
            response_cn = ifft2(responsef_cn, 'symmetric');
            %compute ISLRs of three indiviual response maps
			islr=compute_islr(response,interp_sz);
            islr_gray=compute_islr(response_gray,interp_sz);
            islr_cn=compute_islr(response_cn,interp_sz);
            ISLR=[islr islr_gray islr_cn];
            %use softmax to normalize the response maps, denote by '_soft'
            response_soft=exp(response);
            response_soft=response_soft/(sum((sum(response_soft))'));
            response_soft_gray=exp(response_gray);
            response_soft_gray=response_soft_gray/(sum((sum(response_soft_gray))'));
            response_soft_cn=exp(response_cn);
            response_soft_cn=response_soft_cn/(sum((sum(response_soft_cn))'));
            %fuse the response maps, denoted by '_all'
            response_all=response_soft.*response_soft_gray.*response_soft_cn;
            
            [row_all, col_all] = find(response_all == max(response_all(:)), 1);
            disp_row_all = mod(row_all - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col_all = mod(col_all - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);            
            
            switch params.interpolate_response
                case 0
                    translation_vec_all = round([disp_row_all, disp_col_all] * featureRatio * currentScaleFactor);
                case 1
                    translation_vec_all = round([disp_row_all, disp_col_all] * currentScaleFactor);
                case 2
                    translation_vec_all = [disp_row_all, disp_col_all];
            end
            
            pos_all = pos_all + translation_vec_all;

        %%%%%%% the first-stage ISLR evaluation %%%%%%%%%%%
            if (sum(ISLR<0.8)>=3)||(sum(ISLR<0.5)>=2)||(sum(ISLR<-2.5)>=1)
                second_stage_flag=1;
            end
            iter = iter + 1;
        end
        
        %%%%%%%% the Second-stage  Siamese networks evaluation %%%%%%%%
        if second_stage_flag == 1 
            im_color = imread(s_frames{frame});
            
            last_rect_all = rect_position_all(frame - 1, :);      
            current_sz_all  = [last_rect_all(4) last_rect_all(3)];
            current_box_all = [pos_all([2,1]) - current_sz_all([2,1])/2, current_sz_all([2,1])];
            
            input_roi_all = get_rois(current_box_all, evaluate_param.imageSz, im_color);            
            input_im_all  = prepare_image(im_color, evaluate_param.imageSz, evaluate_param.pixel_means);
            
            input_blobs_all    = cell(2, 1);
            input_blobs_all{1} = input_im_all;
            net.blobs('rois').reshape([5, size(input_roi_all, 1)]);
            input_blobs_all{2} = input_roi_all';
            
            blobs_out_all = net.forward(input_blobs_all);
            tfeat_all     = squeeze(blobs_out_all{1});
            
            %compute evaluation score
            score_all     = tfeat_all' * evaluate_param.firstframe_feat;
            
            score_all = max(score_all(:));           
            if score_all < evaluate_param.threshold_all
                verification_result = false;
            end
        end
        
        if ~ verification_result
            % Rectification
            pos_all = rectify_tracker(im, pos_all, current_sz_all, evaluate_param);
        else
            second_stage_flag=0;
        end
        %%%%%%%%         end of the Second-stage    %%%%%%%%
        
        %scale search
        if params.number_of_scales > 0
            
            %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im,pos_all,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
            xsf = fft(xs,[],2);
            
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + params.lambda);       
            interp_scale_response = ifft( resizeDFT(scale_responsef, params.number_of_interp_scales), 'symmetric');
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
            %adjust to make sure the size is not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end
    end
    
    %Compute coefficients for the translation filter
    [xl_npca, xl_pca] = get_subwindow(im, pos_all, sz, currentScaleFactor);
    [xl_npca_gray,xl_pca_gray]=get_subwindow_gray(im, pos_all, sz, currentScaleFactor);
    [xl_npca_cn, xl_pca_cn] = get_subwindow_cn(im, pos_all, sz, cn_non_compressed_features, cn_compressed_features, w2c, currentScaleFactor);
    
    if frame == 1
        h_num_pca = xl_pca;
        h_num_npca = xl_npca;
        h_num_pca_gray = xl_pca_gray;
        h_num_npca_gray = xl_npca_gray;   
        h_num_pca_cn = xl_pca_cn;
        h_num_npca_cn = xl_npca_cn;          
        
        % set number of compressed dimensions to maximum if too many
        params.num_compressed_dim = min(params.num_compressed_dim, size(xl_pca, 2));
        cn_num_compressed_dim = min(cn_num_compressed_dim, size(xl_pca_cn, 2));

    else
        h_num_pca = (1 - params.interp_factor) * h_num_pca + params.interp_factor * xl_pca;
        h_num_npca = (1 - params.interp_factor) * h_num_npca + params.interp_factor * xl_npca;
        h_num_pca_gray = (1 - params.interp_factor) * h_num_pca_gray + params.interp_factor * xl_pca_gray;
        h_num_npca_gray = (1 - params.interp_factor) * h_num_npca_gray + params.interp_factor * xl_npca_gray;
        h_num_pca_cn = (1 - params.interp_factor) * h_num_pca_cn + params.interp_factor * xl_pca_cn;
        h_num_npca_cn = (1 - params.interp_factor) * h_num_npca_cn + params.interp_factor * xl_npca_cn;
    end
    data_matrix = h_num_pca;
    data_matrix_cn = h_num_pca_cn;
    
    [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
    projection_matrix = pca_basis(:, 1:params.num_compressed_dim);
    [pca_basis_cn, ~, ~] = svd(data_matrix_cn' * data_matrix_cn);  
    projection_matrix_cn = pca_basis_cn(:, 1:cn_num_compressed_dim);  
        
    hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
    hf_num = bsxfun(@times, yf, conj(hf_proj)); 
    hf_proj_gray = fft2(feature_projection(h_num_npca_gray, h_num_pca_gray, 1, cos_window));
    hf_num_gray = bsxfun(@times, yf, conj(hf_proj_gray));
    hf_proj_cn = fft2(feature_projection_cn(h_num_npca_cn, h_num_pca_cn, projection_matrix_cn, cos_window));
    hf_num_cn = bsxfun(@times, yf, conj(hf_proj_cn));     

    xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
    new_hf_den = sum(xlf .* conj(xlf), 3);
    xlf_gray = fft2(feature_projection(xl_npca_gray, xl_pca_gray, 1, cos_window));
    new_hf_den_gray = sum(xlf_gray .* conj(xlf_gray), 3);
    xlf_cn = fft2(feature_projection_cn(xl_npca_cn, xl_pca_cn, projection_matrix_cn, cos_window));
    new_hf_den_cn = sum(xlf_cn .* conj(xlf_cn), 3);    

        
    if frame == 1
        hf_den = new_hf_den;
        hf_den_gray = new_hf_den_gray;
        hf_den_cn = new_hf_den_cn;
    else
        hf_den = (1 - params.interp_factor) * hf_den + params.interp_factor * new_hf_den;
        hf_den_gray = (1 - params.interp_factor) * hf_den_gray + params.interp_factor * new_hf_den_gray;
        hf_den_cn = (1 - params.interp_factor) * hf_den_cn + params.interp_factor * new_hf_den_cn;
    end
       
    %Compute coefficents for the scale filter
    if params.number_of_scales > 0
        
        %create a new feature projection matrix
        [xs_pca, xs_npca] = get_scale_subwindow(im, pos_all, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
        if frame == 1
            s_num = xs_pca;
        else
            s_num = (1 - params.interp_factor) * s_num + params.interp_factor * xs_pca;
        end
        bigY = s_num;
        bigY_den = xs_pca;
        
        if max_scale_dim
            [scale_basis, ~] = qr(bigY, 0);
            [scale_basis_den, ~] = qr(bigY_den, 0);      
        else
            [U,~,~] = svd(bigY,'econ');
            scale_basis = U(:,1:s_num_compressed_dim);     
        end
        scale_basis = scale_basis';
        
        % create the filter update coefficients
        sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
        sf_num = bsxfun(@times,ysf,conj(sf_proj));    
        
        xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);
               
        if frame == 1
            sf_den = new_sf_den;
        else
            sf_den = (1 - params.interp_factor) * sf_den + params.interp_factor * new_sf_den;
        end
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    % save position 
    rect_position_all(frame, :) = [pos_all([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];   
    
    time = time+toc();
    % visualization
    if visualization == 1
        rect_position_vis_all = [pos_all([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; 
        if frame == 1
            figure;
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle_all = rectangle('Position',rect_position_vis_all, 'EdgeColor','red', 'LineWidth', 2); 
            text_handle = text(10, 10, int2str(frame), 'FontSize', 18);
            set(text_handle, 'color', [0 1 1]);
        else
            try
                set(im_handle, 'CData', im)
                set(rect_handle_all, 'Position', rect_position_vis_all)
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        drawnow
    end
end
fps = frame / time;
results.fps = fps;
results.type = 'rect';
results.res=rect_position_all;
end