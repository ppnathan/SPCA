function [result_labels class_acc overall_acc] = NS(test_hist, train_hist, ground_truth)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compute the nearest subspace
% test_hist : testing histogram
% support : the support of each subspace
% ground_truth : ground truth for test_hist
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[feat_dim num_imgs_per_obj num_objects ]= size(test_hist);
% train_hist = train_hist - repmat(mean(train_hist, 2), 1, size(train_hist, 2));
% test_hist = test_hist - repmat(mean(test_hist, 2), 1, size(test_hist, 2));

result_labels = zeros(1, num_imgs_per_obj, num_objects);
for i = 1:num_objects
    X = train_hist(:, :, i);
    SS_mat(:, :, i) = eye(feat_dim) - (X/(X'*X))*X';
end


for i = 1:num_objects
    dist_subspace = zeros(num_objects, num_imgs_per_obj);
    for k = 1:num_objects
        diff = SS_mat(:, :, k)*test_hist(:, :, i);
        dist_subspace(k, :) = sqrt(sum(diff.^2, 1));
    end
    
    [min_val_between_objs min_id_between_objs] = min(dist_subspace);
    result_labels(:, :, i) = min_id_between_objs;

end

error = result_labels - ground_truth;
class_acc = sum(abs(error)==0, 2)./num_imgs_per_obj;
overall_acc = sum(sum(abs(error)==0))./(num_imgs_per_obj*num_objects);

end
