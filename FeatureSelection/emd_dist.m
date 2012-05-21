function hist_dist = emd_dist(test_hist_vec, train_hist)

[dim_feat n_imgs n_objs] = size(train_hist);
hist_dist = zeros([1 n_imgs n_objs]);

for i = 1:n_objs
    for j = 1:n_imgs
        hist_dist(:, j, i) = emdL1( test_hist_vec, train_hist(:, j, i) );
    end
end

end