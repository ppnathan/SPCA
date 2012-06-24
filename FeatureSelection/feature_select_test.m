close all; clear all;

load raw_features.mat;
load spca_supports.mat;

BMW_objects = {'bowles'; 'california'; 'campanile'; 'eastasianlibrary'; 'evans'; 'foothill'; 'garden'; ...
     'haas'; 'hearstgym'; 'hertzmorrison'; 'hilgard'; 'hmc'; 'logcabin'; 'mainlibrary'; 'musiclibrary'; ...
     'parkinglot'; 'sathergate'; 'sproul'; 'vlsb'; 'wurster'};
test_cameras_id = {'00', '01', '03', '04'};
test_images_id = {'0000'; '0001'; '0002';'0003'; '0004';'0005'; '0006';'0007'; '0008';'0009'; '0010';'0011'; '0012';'0013'; '0014'; '0015'};
data_dir = '../../opencv/BMW';

num_test_objects = length(BMW_objects);
num_test_cameras = length(test_cameras_id);
num_test_images_per_camera = length(test_images_id);
num_total_test_images = num_test_objects*num_test_cameras*num_test_images_per_camera;

HIST_DIM = 1000;

load test_histogram;
%% testing with baseline method (with all features)
% test_histogram = zeros(HIST_DIM, num_test_images_per_camera*num_test_cameras, num_test_objects);
% for i =1:num_test_objects
%     cnt_img = 1;
%     for j = 1:num_test_cameras
%         for k = 1:num_test_images_per_camera
%              [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{i}, test_cameras_id{j}, test_images_id{k}, 0);
%              feat_cata = zeros(size(locs, 1), 1);
%              for p = 1:size(locs, 1)
%                  AT = vl_hikmeanspush(train_tree, uint8(desc(p, :)'*255));
%                  if HIST_DIM == 1000
%                      feat_cata(p) = (AT(1)-1)*100 + (AT(2)-1)*10 + AT(3);
%                  elseif HIST_DIM == 10000
%                      feat_cata(p) = (AT(1)-1)*1000 + (AT(2)-1)*100 + AT(3)*10 + AT(4);   
%                  else
%                      fprintf('Error: Unsupported HIST_DIM: %d\n', HIST_DIM);
%                      return
%                  end
%                  
%              end
%              bins = 1:1:HIST_DIM;
%              test_histogram(:, cnt_img, i) = histc(feat_cata, bins)';
%              fprintf('i = %d / %d, j = %d / %d, k = %d / %d\n', i, num_test_objects, j, num_test_cameras, k, num_test_images_per_camera)
%              cnt_img = cnt_img+1;
%         end
%     end
% end
% 
% test_histogram = test_histogram.*repmat(tdf, [1 size(test_histogram, 2) size(test_histogram, 3)]);
% save test_histogram.mat test_histogram;

% test_histogram is an D x N x C, where D := dimension of histogram
% (i.e. 1000, 10000), N := test image instance, C := Object Class

ground_truth(1, 1, :) = 1:1:num_test_objects;
ground_truth = repmat(ground_truth, [1 num_test_images_per_camera*num_test_cameras]);

tic;
disp('== Testing baseline...');
[result_labels_bl class_acc_bl overall_acc_bl] = NN_1(test_histogram, train_histogram, ground_truth, 'l1');
%[result_labels_bl class_acc_bl overall_acc_bl] = NS(test_histogram, train_histogram, ground_truth); % CRASHED HERE
fprintf('Baseline Overall Accuracy: %f\n', overall_acc_bl);
time1 = toc;
disp('');
%% testing with infomative features
support_id = find(total_support ==1);

test_histogram_sp = test_histogram(support_id, :, :);
train_histogram_sp = train_histogram(support_id, :, :);
% support_sp = support(support_id, :);
tic;
disp('== Testing SPCA...');
[result_labels_spca class_acc_spca overall_acc_spca] = NN_1(test_histogram_sp, train_histogram_sp, ground_truth, 'l1');
%[result_labels_spca class_acc_spca overall_acc_spca] = NS(test_histogram_sp, train_histogram_sp, ground_truth);
fprintf('SPCA Overall Accuracy: %f\n', overall_acc_spca);
time2 = toc;
