function [hist_dist] = chi_square_dist(test_hist_vec, train_hist)

diff_hist = train_hist - repmat(test_hist_vec, [1 size(train_hist, 2) size(train_hist, 3)]);
sum_hist = train_hist + repmat(test_hist_vec, [1 size(train_hist, 2) size(train_hist, 3)]) +1e-8;
hist_dist = 0.5*sum(diff_hist.^2./sum_hist, 1);
% hist_dist = sum(abs(diff));

end