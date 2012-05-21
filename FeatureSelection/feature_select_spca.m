close all; clear all;

load raw_features.mat;
addpath('GPower');

BMW_objects = {'bowles'; 'california'; 'campanile'; 'eastasianlibrary'; 'evans'; 'foothill'; 'garden'; ...
     'haas'; 'hearstgym'; 'hertzmorrison'; 'hilgard'; 'hmc'; 'logcabin'; 'mainlibrary'; 'musiclibrary'; ...
     'parkinglot'; 'sathergate'; 'sproul'; 'vlsb'; 'wurster'};


%% run SPCA
num_pc = 2;
rho = 0.3*ones(1, num_pc);
support = zeros(1000, length(BMW_objects));
for i = 1:length(BMW_objects)
    i
    feat_cov(:, :, i) = cov(train_histogram(:, :, i)');
    
% SAFE variable elimination for SPCA
%     remain_id = find(diag(feat_cov(:, :, i)) > rho)';
%     feat_cov_new = feat_cov(remain_id, remain_id, i);
    remain_id = 1:1:1000;
    histogram_new = train_histogram(remain_id, :, i);
    norm_hist_new = sqrt(sum(histogram_new.^2, 1));
    hist_new_dim = size(histogram_new, 1);
    for k = 1:hist_new_dim
        histogram_new(k, :) = histogram_new(k, :)./norm_hist_new;
    end
    histogram_new = histogram_new - repmat(mean(histogram_new, 2), 1, size(histogram_new, 2));
% Sparse PCA
    tic;
%     [x, numIter] = SPCA_ALM(feat_cov_new, rho);
%     one eigenvector    
%     x = GPower(histogram_new', rho, num_pc, 'l1', 0);
%     y = x.*(abs(x)>1e-3);

%     two eigenvectors
    x = GPower(histogram_new', rho.^2, num_pc, 'l0', 0);
    y = sum(abs(x)>1e-3, 2);

%     one eigenvector block
%     mu = [1:num_pc].^(-1);
%     x = GPower(histogram_new', rho, num_pc, 'l1', 1, mu);
%     y = sum(abs(x)>1e-3, 2);
    
    escape_time(i) = toc;
    
    support(remain_id, i) = y;

end
    total_support = (sum(abs(support), 2) >0);
    num_total_support = sum(total_support);

save spca_supports.mat support total_support num_total_support;


%% visualize the result

train_camera_id = '02';
train_image_id = {'0000'; '0002';'0004';'0006';'0008';'0010';'0012';'0014';};
data_dir = '../../opencv/BMW';
[locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{3}, train_camera_id, train_image_id{2}, 0);

pathToImag = [data_dir '/' BMW_objects{3} '/' train_camera_id '/' train_image_id{2} '.jpg'];
% Plot the image and the features
im = imread(pathToImag);
figure(1); clf; imshow(im); hold on; plot(locs(:,1), locs(:,2), 'rx');

figure(2);
imshow(im); hold on;

for i =1:size(desc, 1);
    hi_labels = vl_hikmeanspush(train_tree, uint8(desc(i, :)'*255));
    numerical_labels = (hi_labels(1)-1)*100+(hi_labels(2)-1)*10+hi_labels(3);
    if (total_support(numerical_labels) > 0)
        plot(locs(i,1), locs(i,2), 'rx', 'markersize', 12);
    end
end