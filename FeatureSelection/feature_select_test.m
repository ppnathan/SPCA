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

load test_histogram;
%% testing with baseline method (with all features)
% for i =1:num_test_objects
%     cnt_img = 1;
%     for j = 1:num_test_cameras
%         for k = 1:num_test_images_per_camera
%              [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{i}, test_cameras_id{j}, test_images_id{k}, 1);
%              feat_cata = zeros(size(locs, 1), 1);
%              for p = 1:size(locs, 1);
%                  AT = vl_hikmeanspush(train_tree, uint8(desc(p, :)'*255));
%                  feat_cata(p) = (AT(1)-1)*100 + (AT(2)-1)*10 + AT(3);
%              end
%              bins = 1:1:1000;
%              test_histogram(:, cnt_img, i) = histc(feat_cata, bins)';
%              cnt_img = cnt_img+1;
%         end
%     end
% end
% 
% test_histogram = test_histogram.*repmat(tdf, [1 size(test_histogram, 2) size(test_histogram, 3)]);
% save test_histogram.mat test_histogram;
tic;
[result_labels_bl class_acc_bl overall_acc_bl] = NN_1(test_histogram, train_histogram, 'l1');
time1 = toc;

%% testing with infomative features
support_id = find(total_support ==1);

test_histogram_sp = test_histogram(support_id, :, :);
train_histogram_sp = train_histogram(support_id, :, :);
tic;
[result_labels_spca class_acc_spca overall_acc_spca] = NN_1(test_histogram_sp, train_histogram_sp, 'l1');
time2 = toc;
