function [result_labels class_acc overall_acc] = NN_1(test_hist, train_hist, ground_truth, distance_type)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compute the nearest neighborhood
% test_hist : testing histogram
% train_hist : training histogram
% distance_type : different types of distance used to measure the distance 
%                 of test histogram and training histogram,
%                 there are 'l1' : l1 norm distance
%                           'int' : intersection distance
%                           'chi_sq' : chi-square distance
%                           'emd' : earth mover's distance
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('EmdL1_v3');

[feat_dim num_imgs_per_obj num_objects ]= size(test_hist);

% normalize the vectors
if (strcmp(distance_type, 'l1') || strcmp(distance_type,'int'))
    norm_test_hist = sum(abs(test_hist), 1);%sqrt(sum(test_hist.^2, 1));
    for i =1:feat_dim
        test_hist(i, :, :) = test_hist(i, :, :)./norm_test_hist;
    end
    norm_train_hist = sum(abs(train_hist), 1);%sqrt(sum(train_hist.^2, 1));
    for i =1:feat_dim
        train_hist(i, :, :) = train_hist(i, :, :)./norm_train_hist;
    end
elseif (strcmp(distance_type, 'chi_sq') || strcmp(distance_type, 'emd'))
    norm_test_hist = sqrt(sum(test_hist.^2, 1));
    for i =1:feat_dim
        test_hist(i, :, :) = test_hist(i, :, :)./norm_test_hist;
    end
    norm_train_hist = sqrt(sum(train_hist.^2, 1));
    for i =1:feat_dim
        train_hist(i, :, :) = train_hist(i, :, :)./norm_train_hist;
    end
end

result_labels = zeros(1, num_imgs_per_obj, num_objects);

for i = 1:num_objects
    
    for j = 1:num_imgs_per_obj
        
        if strcmp(distance_type, 'l1')
            hist_dist = l1_dist(test_hist(:, j, i), train_hist);
        elseif strcmp(distance_type, 'int')
            hist_dist = intersection_dist(test_hist(:, j, i), train_hist);
        elseif strcmp(distance_type, 'chi_sq')
            hist_dist = chi_square_dist(test_hist(:, j, i), train_hist);
        elseif strcmp(distance_type, 'emd')
            hist_dist = emd_dist(test_hist(:, j, i), train_hist);
        end
        
        [min_val_same_obj  min_id_same_obj] = min(hist_dist);
        [min_val_between_objs min_id_between_objs] = min(min_val_same_obj);
        result_labels(:, j, i) = min_id_between_objs;
    end
end

% ground_truth(1, 1, :) = 1:1:num_objects;
% ground_truth = repmat(ground_truth, [1 num_imgs_per_obj]);

error = result_labels - ground_truth;
class_acc = sum(abs(error)==0, 2)./num_imgs_per_obj;
overall_acc = sum(sum(abs(error)==0))./(num_imgs_per_obj*num_objects);

end