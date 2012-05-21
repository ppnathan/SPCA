function [hist_dist] = intersection_dist(test_hist_vec, train_hist)

min_val = min(train_hist, repmat(test_hist_vec, [1 size(train_hist, 2) size(train_hist, 3)]));
hist_dist = ones([1 size(train_hist, 2) size(train_hist, 3)]) - sum(min_val, 1);
% hist_dist = sum(abs(diff));

end