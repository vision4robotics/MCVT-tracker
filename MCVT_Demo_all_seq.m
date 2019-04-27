function MCVT_Demo_all_seq(save_dir)  
    close all;
    clc;
    
    %% **Need to change**
    where_is_your_groundtruth_folder = 'E:\Final\UAV123_10fps\anno\UAV123_10fps';      %the groundturth folder   
    where_is_your_UAV123_database_folder = 'E:\Final\UAV123_10fps\data_seq\UAV123_10fps';     %the sequences folder
    tpye_of_assessment = 'UAV123_10fps';                                   
    tracker_name = 'MCVT';                                                  
    
    %% Read all video names using grouthtruth.txt
    ground_truth_folder = where_is_your_groundtruth_folder;                
    dir_output = dir(fullfile(ground_truth_folder, '\*.txt'));             
    contents = {dir_output.name}';
    all_video_name = {};
    for k = 1:numel(contents)
        name = contents{k}(1:end-4);                                       
        all_video_name{end+1,1} = name;                                    
    end
    dataset_num = length(all_video_name);                                  
    type = tpye_of_assessment;                                             

    %%
    for dataset_count = 1:dataset_num  
        video_name = all_video_name{dataset_count};                        
        database_folder = where_is_your_UAV123_database_folder;            
        seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type); 

        % main function
        result  =  run_MCVT(seq);                                           

        % save results
        results = cell(1,1);                                               
        results{1} = result;
        results{1}.len = seq.len;
        results{1}.startFrame = seq.st_frame;
        results{1}.annoBegin = seq.st_frame;
        
        % save results to specified folder
        if nargin < 1
            save_dir = '.\MCVT\';                              
        end 
        save_res_dir = [save_dir, tracker_name, '_results\'];       %the results saving folder         
        save_pic_dir = [save_res_dir, 'res_picture\'];              %the precision plots saving folder       
        if ~exist(save_res_dir, 'dir')
            mkdir(save_res_dir);
            mkdir(save_pic_dir);
        end 
        save([save_res_dir, video_name, '_', tracker_name], 'results');    

        % plot precision figure
        show_visualization = 1;                                            
        results{1}.res=results{1}.res;                                 
        precision_plot_v1(results{1}.res, seq.ground_truth, video_name, save_pic_dir, show_visualization); 

        close all;
    end