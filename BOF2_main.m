
% Zack Beucler
% COM322 Final Project
% Classifying images of cars by brand (Ford, Nissian, Honda, or Toyota)
% BOF2_main.m
% This script uses bag of features and an SVM to classify images. 
    % I used the trainImageCategoryClassifier function to create the model


%% load and label all images in a datastore to save time and storage
clear;
imds = imageDatastore('Datasets/Full', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


%% visualize the data
visualizeData(imds)


%% create testing and training datasets
% training has 700 from each class (2800 total) and testing has 400 from each class (1600 total)
[training_set, test_set] = imds.splitEachLabel(700,400,'randomize',true);


%% bag of features
bag = bagOfFeatures(training_set, 'VocabularySize', 250, 'PointSelection', 'Detector');
data = double(encode(bag, training_set));

%% Visualize bag of features data
visualizeBOFData(training_set, bag)


%% Train the classifier using trainImageCategoryClassifier
categoryClassifier = trainImageCategoryClassifier(training_set, bag);
cm = evaluate(categoryClassifier, training_set);
fprintf("Accuracy over the training set: %2.2f %%\n", mean(diag(cm))*100);


%% Show predictions for individual images from the test set
num_of_images = 50; 
test_set_size = numel(test_set.Files);
total_correct = 0;
for a = 1:num_of_images
    rand_num = randi([0 test_set_size]); % get valid random number
    img = readimage(test_set, rand_num); % load image
    GT_label = string(test_set.Labels(rand_num)); % get ground truth for that image
    [labelIdx, scores] = predict(categoryClassifier, img); % perform prediction
    pred_label = string(categoryClassifier.Labels(labelIdx)); % comes as 1x1 cell so i casted it as string
    fprintf("\nPredicted brand: %s \nActual brand: %s\n", pred_label, GT_label);
    if GT_label == pred_label
        total_correct = total_correct+1;
    end
    coolImShow(img, pred_label, GT_label)
end
fprintf("\nPercent correct: %2.2f %%\n", (total_correct/num_of_images)*100);


%% evaluate the test set and display the accuracy
confMatrix = evaluate(categoryClassifier,test_set);
fprintf("Accuracy over the test set: %2.2f %%\n", mean(diag(confMatrix))*100);


%% helper functions

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
    
function visualizeBOFData(training_set, bag)
    img = readimage(training_set, numel(training_set.Files));
    visData = encode(bag, img);
    figure(1);
    subplot(2,2,[1 2])
    imshow(img)
    subplot(2,2,[3 4])
    bar(visData)
    title('Visual word occurrences')
    xlabel('Visual word index')
    ylabel('Frequency of occurrence')
end