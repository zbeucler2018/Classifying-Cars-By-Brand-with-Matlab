
% Zack Beucler
% COM322 Final Project
% Classifying images of cars by brand (Ford, Nissian, Honda, or Toyota)
% HOG_main.m
% This script uses hog features as data to train a model to classify the images.


%% load and label all images in a datastore to save time and storage
clear;
imds = imageDatastore('Datasets/Full', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


%% visualize the data
visualizeData(imds)


%% create testing and training datasets
% training has 700 of each label(2800 total) and test has 400 of each label(1600 total)
[training_set, test_set] = imds.splitEachLabel(700,400,'randomize',true);


%% Visualize HOG features of random image in training set
img = readimage(training_set, 15); % get the 15th image in the training set
visualizeHOGData(img)
cell_size = [4 4];
hogFeatureSize = length(extractHOGFeatures(img,'CellSize',cell_size)); % get the length of the feature vector


%% Train classifer
numImages = numel(training_set.Files); % get total # of images in training set
trainingFeatures = zeros(numImages, hogFeatureSize, 'single'); % empty matrix to store hog features of training set
for i = 1:numImages
    img = readimage(training_set, i); % get image from training set
    img = rgb2gray(img); % convert to grayscale because i didnt think color in the imgs were needed
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cell_size); % get HOG features
end
fprintf("\nFinished getting HOG features\n");
trainingLabels = training_set.Labels; % get the labels from the training set
classifier = fitcecoc(trainingFeatures, trainingLabels); % train classifier
fprintf("\nFinished training\n");


%% make predictions for entire training set
[t_Features, t_Labels] = helperExtractHOGFeaturesFromImageSet(training_set, hogFeatureSize, cell_size); % get hog features for every image in testset
training_GT = training_set.Labels; % get the ground truth
predictedLabels = predict(classifier, t_Features); % run the prediction
correctPredictions = (predictedLabels == training_GT); % get the total amount of correct predictions
validationAccuracy = sum(correctPredictions)/length(predictedLabels)*100; % accuracy for entire test set
fprintf("\nAccuracy on training set: %2.2f %%\n", validationAccuracy);


%% Show predictions for individual images from the test set
num_of_images = 50;
test_set_size = numel(test_set.Files);
total_correct = 0;
for a = 1:num_of_images
    rand_num = randi([1 test_set_size]); % get valid random number
    img = readimage(test_set, rand_num); % load image
    img_g = rgb2gray(img); % convert to grayscale
    img_features = extractHOGFeatures(img_g, 'CellSize', cell_size); % extract HOG features
    pred_brand = predict(classifier, img_features); % preform prediction
    actual_brand = test_set.Labels(rand_num); % get the ground truth
    fprintf("\nPredicted brand: %s \nActual brand: %s\n", pred_brand, actual_brand);
    if pred_brand == actual_brand
        total_correct = total_correct+1;
    end
    coolImShow(img, pred_brand, actual_brand)
end
fprintf("\nPercent correct: %2.2f %%\n", (total_correct/num_of_images)*100);


%% make predictions for entire test set
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(test_set, hogFeatureSize, cell_size); % get hog features for every image in testset
test_GT = test_set.Labels; % get the ground truth
predictedLabels = predict(classifier, testFeatures); % run the prediction
correctPredictions = (predictedLabels == test_GT); % get the total amount of correct predictions
validationAccuracy = sum(correctPredictions)/length(predictedLabels)*100; % accuracy for entire test set
fprintf("\nAccuracy on test set: %2.2f %%\n", validationAccuracy);


%% helper functions

function coolImShow(img, pred_label, GT_label)
    pred_label = string(pred_label); GT_label = string(GT_label);
    position = [1 1; 1 35];
    sec_color = "red";
    if pred_label == GT_label
        sec_color = "green";
    end
    color = {"green", sec_color};
    prediction_text = strcat("Predicted Label: ", pred_label);
    GT_text = strcat("Ground Truth: ", GT_label);
    RGB = insertText(img, position, [GT_text prediction_text], 'FontSize', 18, 'BoxColor', color, 'BoxOpacity', 0.4, 'TextColor','black');
    imshow(RGB);pause(1);
end


function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
    % Extract HOG features from an imageDatastore.
    setLabels = imds.Labels;
    numImages = numel(imds.Files);
    features  = zeros(numImages,hogFeatureSize,'single');
    % Process each image and extract features
    for j = 1:numImages
        img = readimage(imds,j);
        img = rgb2gray(img);
        features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
    end
end

function visualizeData(imds)
    tbl = countEachLabel(imds);
    categories = tbl.Label; tbl
    sample = splitEachLabel(imds, 1, 'randomize'); % get a single random image from each label
    for ii = 1:4 % 4 different brands (ford, honda, nissan, toyota)
        subplot(2,2,ii);
        imshow(sample.readimage(ii));
        title(char(sample.Labels(ii)));
    end
end

function visualizeHOGData(img)
    % Extract HOG features and HOG visualization
    [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
    % Show the original image
    figure; subplot(2,2,[1 2]); imshow(img);
    % Visualize the HOG features
    subplot(2,2,[3 4]);
    plot(vis4x4);
    title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
end


