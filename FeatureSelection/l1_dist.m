function [hist_dist] = l1_dist(test_hist_vec, train_hist)

diff = train_hist - repmat(test_hist_vec, [1 size(train_hist, 2) size(train_hist, 3)]);
hist_dist = sum(abs(diff));

end