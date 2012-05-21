function [locs, desc, surfFeatures] = ParseSURFFile(rootImageDir, rootFeatureDBDir, object, camera, imageID, isShowImage)

% this function is used to parse the SURF feature file
% rootImageDir = The directory with the BMW database
% rootFeatureDBDir = The directory with the SURF features. i.e. BMW_SURF
% object : string with the object category. For ex., campinile, bowles, etc.
% camera : string with the camera number. For ex., 00, 01, 02, 03, 04
% imageID : string with the image number. For ex., 0000, 0001, ...
% isShowImage : plot the image and its SURF features


pathToFile = [rootFeatureDBDir '/' object '/' camera '/' imageID '_surf.txt'];
pathToImag = [rootImageDir '/' object '/' camera '/' imageID '.jpg'];

% Open tmp.key and check its header
g = fopen(pathToFile, 'r');
[header, count] = fscanf(g, '%d %d', [1 2]);
num = header(1);
len = header(2);

% Create the two output matrices (use known size for efficiency)
locs = double(zeros(num, 2));
descriptors = double(zeros(num, len));
siftFeatures = [];

% Parse file
for i = 1:num
    [vector, count] = fscanf(g, '%f %f %f %f %f %f', [1 6]); %row col scale ori
    if count ~= 6
        error('Invalid keypoint file format');
    end
    %locs(i, :) = vector(1, :);

    [descrip, count] = fscanf(g, '%f', [1 len]);
    if (count ~= len)
        error('Invalid keypoint file value.');
    end

    % Normalize each input vector to unit length
    descriptor = descrip / sqrt(sum(descrip.^2));
    desc(i,:) = descriptor;
    locs(i, :) = vector(1, [1 2]);
    
    surfFeatures(i).location = vector(1,[1 2]);
    surfFeatures(i).laplacian = vector(1,3);
    surfFeatures(i).size = vector(1,4);
    surfFeatures(i).dir = vector(1, 5);
    surfFeatures(i).hessian = vector(1, 6);
    surfFeatures(i).descriptor = descriptor;
    
end
% locs = locs(:,[2 1]);
fclose(g);

% Plot the image and the features
if isShowImage
    im = imread(pathToImag);
    figure(1); clf; imshow(im); hold on; plot(locs(:,1), locs(:,2), 'rx');
end


end