close all; clear all;


BMW_objects = {'bowles'; 'california'; 'campanile'; 'eastasianlibrary'; 'evans'; 'foothill'; 'garden'; ...
     'haas'; 'hearstgym'; 'hertzmorrison'; 'hilgard'; 'hmc'; 'logcabin'; 'mainlibrary'; 'musiclibrary'; ...
     'parkinglot'; 'sathergate'; 'sproul'; 'vlsb'; 'wurster'};
% [locs, desc, surfFeatures] = ParseSURFFile('BMW', 'BMW_SURF', BMW_objects{3}, '01', '0000');

HIST_DIM = 10000;

%% collect features from all the training landmarks
train_camera_id = '02';
train_images_id = {'0000'; '0002'; '0004'; '0006'; '0008'; '0010'; '0012'; '0014'};
data_dir = '../../opencv/BMW';
train_surf = [];
train_locs = [];
num_objects = length(BMW_objects);
num_img_each_object = length(train_images_id);
num_features = zeros(num_objects, num_img_each_object);
%SURF_DIM = 128;
%train_surf = zeros(100, SURF_DIM);
for i = 1:num_objects
    for j = 1:num_img_each_object
        fprintf('i=%d/%d j=%d/%d\n', i, num_objects, j, num_img_each_object);
        [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{i}, train_camera_id, train_images_id{j}, 0);
        num_features(i, j) = size(locs, 1);
        %disp(size(desc));
        train_surf = [train_surf; desc];
        train_locs = [train_locs; [i*ones(num_features(i, j), 1)  j*ones(num_features(i, j), 1) locs]];
        
    end
end

% train_surf is an Nx128 matrix, where N is the total number of SURF
% features across all classes/training instances.


%% create a hierachical k-mean tree

K = 10;
nleaves = HIST_DIM;
% uint8_surf = uint8(train_surf'*255);
% input to vl_hikmeans is 128xN, where N is the total number of SURF
% features.

[train_tree, A] = vl_hikmeans(uint8(train_surf'*255), K, nleaves, 'method', 'elkan') ;
if HIST_DIM == 1000
    train_labels = (A(1, :)-1)*(K^2) + (A(2, :)-1)*K + A(3, :);
elseif HIST_DIM == 10000
    train_labels = (A(1, :)-1)*(K^3) + (A(2, :)-1)*K^2 + (A(3, :)-1)*K + A(4, :);
else
    fprintf('Error, unsupported HIST_DIM: %d\n', HIST_DIM);
    return
end
% train_labels is a 1xN vector, where N is the total number of SURF
% features. train_labels(i) := which leaf node SURF vector i lands on.

% [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{1}, train_camera_id, train_images_id{1});
% AT  = vl_hikmeanspush(train_tree, uint8(desc(1, :)'*255)) ;

%% create histogram


bins = 1:1:HIST_DIM;
start = 1;
train_histogram = zeros(HIST_DIM, num_img_each_object, num_objects);
for i = 1:num_objects;
    for j = 1:num_img_each_object
        train_histogram(:, j, i) = histc(train_labels(1, start:start+num_features(i, j)-1), bins)';
        start = start+num_features(i, j);
    end
end

% compute tf*idf
num_image = num_objects*num_img_each_object;
num_doc_feat_appear = sum(reshape(train_histogram, nleaves, num_image) >0, 2 );
tdf = log((num_image+1)./(1+num_doc_feat_appear));

train_histogram = train_histogram.*repmat(tdf, [1 num_img_each_object num_objects]);

save raw_features.mat train_histogram train_labels train_locs num_features train_tree tdf;

