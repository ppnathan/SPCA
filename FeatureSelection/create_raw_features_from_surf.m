close all; clear all;


BMW_objects = {'bowles'; 'california'; 'campanile'; 'eastasianlibrary'; 'evans'; 'foothill'; 'garden'; ...
     'haas'; 'hearstgym'; 'hertzmorrison'; 'hilgard'; 'hmc'; 'logcabin'; 'mainlibrary'; 'musiclibrary'; ...
     'parkinglot'; 'sathergate'; 'sproul'; 'vlsb'; 'wurster'};
% [locs, desc, surfFeatures] = ParseSURFFile('BMW', 'BMW_SURF', BMW_objects{3}, '01', '0000');

%% collect features from all the training landmarks
train_camera_id = '02';
train_images_id = {'0000'; '0002'; '0004'; '0006'; '0008'; '0010'; '0012'; '0014'};
data_dir = '../../opencv/BMW';
train_surf = [];
train_locs = [];
num_objects = length(BMW_objects);
num_img_each_object = length(train_images_id);

for i = 1:num_objects
    for j = 1:num_img_each_object
        [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{i}, train_camera_id, train_images_id{j}, 1);
        num_features(i, j) = size(locs, 1);
        train_surf = [train_surf; desc];
        train_locs = [train_locs; [i*ones(num_features(i, j), 1)  j*ones(num_features(i, j), 1) locs]];
        
    end
end


%% create a hierachical k-mean tree

K = 10;
nleaves = 1000;
% uint8_surf = uint8(train_surf'*255);
[train_tree, A] = vl_hikmeans(uint8(train_surf'*255), K, nleaves, 'method', 'elkan') ;
train_labels = (A(1, :)-1)*(K^2) + (A(2, :)-1)*K + A(3, :);

% [locs, desc, surfFeatures] = ParseSURFFile(data_dir, data_dir, BMW_objects{1}, train_camera_id, train_images_id{1});
% AT  = vl_hikmeanspush(train_tree, uint8(desc(1, :)'*255)) ;

%% create histogram


bins = 1:1:1000;
start = 1;
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

