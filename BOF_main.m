
% Zack Beucler
% COM322 Final Project
% Classifying images of cars by brand (Ford, Nissian, Honda, or Toyota)
% BOF_main.m
% This script uses bag of features and a medium gaussian model to classify images. 
    % I used the classification learner to make the model and then exported
    % the training process into a function


%% load and label all images in a datastore to save time and storage
clear;
imds = imageDatastore('Datasets/Full', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


%% visualize the data
visualizeData(imds) 


%% create testing and training datasets
% training has 700 of each class (2800 total) and testing has 400 of each class (1400 total)
[training_set, test_set] = imds.splitEachLabel(700,400,'randomize',true);


%% bag of features
bag = bagOfFeatures(training_set, 'VocabularySize', 250, 'PointSelection', 'Detector');
data = double(encode(bag, training_set));


%% visualize bag of features for single image
visualizeBOFData(training_set, bag);


%% create table with features to give to the classification learner
carImageData = array2table(data);
carImageData.Brand = training_set.Labels; % create a new col called 'Brand' and populate it w the labels from the training set


%% Training the Model
% the model I picked was the medium gaussian SVM which had the highest validation accuracy
% when triaing, I used the holdout validation option and I chose to holdout 25%
[trainedClassifier, validationAccuracy] = trainClassifier(carImageData); 
fprintf("\nAccuracy of model on training set: %2.2f %%\n", validationAccuracy*100);


%% Evaluate on individual images in test set
num_of_images = 50;
total_correct = 0;
for w = 1:num_of_images
    rand_num = randi([1 numel(test_set.Files)]); % get a random int between 0 and the size of the test set
    img = readimage(test_set, rand_num); % load random image
    GT_label = test_set.Labels(rand_num); % get the GT for the image
    encoded_img = double(encode(bag, img)); % encode the image
    img_data = array2table(encoded_img, 'VariableNames', trainedClassifier.RequiredVariables); % convert to table
    predicted_label = trainedClassifier.predictFcn(img_data); % perform prediction
    fprintf("\nPredicted brand: %s \nActual brand: %s\n", predicted_label, GT_label);
    if predicted_label == GT_label
        total_correct = total_correct+1;
    end
    coolImShow(img, predicted_label, GT_label);
end
fprintf("\nPercent Correct: %2.2f %%\n", (total_correct/num_of_images)*100)


%% evaluate on entire test set
test_car = double(encode(bag, test_set));
test_car_data = array2table(test_car, 'VariableNames', trainedClassifier.RequiredVariables);
test_GT = test_set.Labels;
predictedOutcome = trainedClassifier.predictFcn(test_car_data);
correctPredictions = (predictedOutcome == test_GT);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome)*100;
fprintf("\nAccuracy on test set: %2.2f %%\n", validationAccuracy);




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


function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
    % [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
    % returns a trained classifier and its accuracy. This code recreates the
    % classification model trained in Classification Learner app. Use the
    % generated code to automate training the same model with new data, or to
    % learn how to programmatically train models.
    %
    %  Input:
    %      trainingData: a table containing the same predictor and response
    %       columns as imported into the app.
    %
    %  Output:
    %      trainedClassifier: a struct containing the trained classifier. The
    %       struct contains various fields with information about the trained
    %       classifier.
    %
    %      trainedClassifier.predictFcn: a function to make predictions on new
    %       data.
    %
    %      validationAccuracy: a double containing the accuracy in percent. In
    %       the app, the History list displays this overall accuracy score for
    %       each model.
    %
    % Use the code to train the model with new data. To retrain your
    % classifier, call the function from the command line with your original
    % data or new data as the input argument trainingData.
    %
    % For example, to retrain a classifier trained with the original data set
    % T, enter:
    %   [trainedClassifier, validationAccuracy] = trainClassifier(T)
    %
    % To make predictions with the returned 'trainedClassifier' on new data T2,
    % use
    %   yfit = trainedClassifier.predictFcn(T2)
    %
    % T2 must be a table containing at least the same predictor columns as used
    % during training. For details, enter:
    %   trainedClassifier.HowToPredict
    % Auto-generated by MATLAB on 16-Dec-2021 15:29:09
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    predictorNames = {'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14', 'data15', 'data16', 'data17', 'data18', 'data19', 'data20', 'data21', 'data22', 'data23', 'data24', 'data25', 'data26', 'data27', 'data28', 'data29', 'data30', 'data31', 'data32', 'data33', 'data34', 'data35', 'data36', 'data37', 'data38', 'data39', 'data40', 'data41', 'data42', 'data43', 'data44', 'data45', 'data46', 'data47', 'data48', 'data49', 'data50', 'data51', 'data52', 'data53', 'data54', 'data55', 'data56', 'data57', 'data58', 'data59', 'data60', 'data61', 'data62', 'data63', 'data64', 'data65', 'data66', 'data67', 'data68', 'data69', 'data70', 'data71', 'data72', 'data73', 'data74', 'data75', 'data76', 'data77', 'data78', 'data79', 'data80', 'data81', 'data82', 'data83', 'data84', 'data85', 'data86', 'data87', 'data88', 'data89', 'data90', 'data91', 'data92', 'data93', 'data94', 'data95', 'data96', 'data97', 'data98', 'data99', 'data100', 'data101', 'data102', 'data103', 'data104', 'data105', 'data106', 'data107', 'data108', 'data109', 'data110', 'data111', 'data112', 'data113', 'data114', 'data115', 'data116', 'data117', 'data118', 'data119', 'data120', 'data121', 'data122', 'data123', 'data124', 'data125', 'data126', 'data127', 'data128', 'data129', 'data130', 'data131', 'data132', 'data133', 'data134', 'data135', 'data136', 'data137', 'data138', 'data139', 'data140', 'data141', 'data142', 'data143', 'data144', 'data145', 'data146', 'data147', 'data148', 'data149', 'data150', 'data151', 'data152', 'data153', 'data154', 'data155', 'data156', 'data157', 'data158', 'data159', 'data160', 'data161', 'data162', 'data163', 'data164', 'data165', 'data166', 'data167', 'data168', 'data169', 'data170', 'data171', 'data172', 'data173', 'data174', 'data175', 'data176', 'data177', 'data178', 'data179', 'data180', 'data181', 'data182', 'data183', 'data184', 'data185', 'data186', 'data187', 'data188', 'data189', 'data190', 'data191', 'data192', 'data193', 'data194', 'data195', 'data196', 'data197', 'data198', 'data199', 'data200', 'data201', 'data202', 'data203', 'data204', 'data205', 'data206', 'data207', 'data208', 'data209', 'data210', 'data211', 'data212', 'data213', 'data214', 'data215', 'data216', 'data217', 'data218', 'data219', 'data220', 'data221', 'data222', 'data223', 'data224', 'data225', 'data226', 'data227', 'data228', 'data229', 'data230', 'data231', 'data232', 'data233', 'data234', 'data235', 'data236', 'data237', 'data238', 'data239', 'data240', 'data241', 'data242', 'data243', 'data244', 'data245', 'data246', 'data247', 'data248', 'data249', 'data250'};
    predictors = inputTable(:, predictorNames);
    response = inputTable.Brand;
    isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    template = templateSVM(...
        'KernelFunction', 'gaussian', ...
        'PolynomialOrder', [], ...
        'KernelScale', 16, ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    classificationSVM = fitcecoc(...
        predictors, ...
        response, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', categorical({'Ford'; 'Honda'; 'Nissan'; 'Toyota'}));
    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    svmPredictFcn = @(x) predict(classificationSVM, x);
    trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));
    % Add additional fields to the result struct
    trainedClassifier.RequiredVariables = {'data1', 'data10', 'data100', 'data101', 'data102', 'data103', 'data104', 'data105', 'data106', 'data107', 'data108', 'data109', 'data11', 'data110', 'data111', 'data112', 'data113', 'data114', 'data115', 'data116', 'data117', 'data118', 'data119', 'data12', 'data120', 'data121', 'data122', 'data123', 'data124', 'data125', 'data126', 'data127', 'data128', 'data129', 'data13', 'data130', 'data131', 'data132', 'data133', 'data134', 'data135', 'data136', 'data137', 'data138', 'data139', 'data14', 'data140', 'data141', 'data142', 'data143', 'data144', 'data145', 'data146', 'data147', 'data148', 'data149', 'data15', 'data150', 'data151', 'data152', 'data153', 'data154', 'data155', 'data156', 'data157', 'data158', 'data159', 'data16', 'data160', 'data161', 'data162', 'data163', 'data164', 'data165', 'data166', 'data167', 'data168', 'data169', 'data17', 'data170', 'data171', 'data172', 'data173', 'data174', 'data175', 'data176', 'data177', 'data178', 'data179', 'data18', 'data180', 'data181', 'data182', 'data183', 'data184', 'data185', 'data186', 'data187', 'data188', 'data189', 'data19', 'data190', 'data191', 'data192', 'data193', 'data194', 'data195', 'data196', 'data197', 'data198', 'data199', 'data2', 'data20', 'data200', 'data201', 'data202', 'data203', 'data204', 'data205', 'data206', 'data207', 'data208', 'data209', 'data21', 'data210', 'data211', 'data212', 'data213', 'data214', 'data215', 'data216', 'data217', 'data218', 'data219', 'data22', 'data220', 'data221', 'data222', 'data223', 'data224', 'data225', 'data226', 'data227', 'data228', 'data229', 'data23', 'data230', 'data231', 'data232', 'data233', 'data234', 'data235', 'data236', 'data237', 'data238', 'data239', 'data24', 'data240', 'data241', 'data242', 'data243', 'data244', 'data245', 'data246', 'data247', 'data248', 'data249', 'data25', 'data250', 'data26', 'data27', 'data28', 'data29', 'data3', 'data30', 'data31', 'data32', 'data33', 'data34', 'data35', 'data36', 'data37', 'data38', 'data39', 'data4', 'data40', 'data41', 'data42', 'data43', 'data44', 'data45', 'data46', 'data47', 'data48', 'data49', 'data5', 'data50', 'data51', 'data52', 'data53', 'data54', 'data55', 'data56', 'data57', 'data58', 'data59', 'data6', 'data60', 'data61', 'data62', 'data63', 'data64', 'data65', 'data66', 'data67', 'data68', 'data69', 'data7', 'data70', 'data71', 'data72', 'data73', 'data74', 'data75', 'data76', 'data77', 'data78', 'data79', 'data8', 'data80', 'data81', 'data82', 'data83', 'data84', 'data85', 'data86', 'data87', 'data88', 'data89', 'data9', 'data90', 'data91', 'data92', 'data93', 'data94', 'data95', 'data96', 'data97', 'data98', 'data99'};
    trainedClassifier.ClassificationSVM = classificationSVM;
    trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2019a.';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    predictorNames = {'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14', 'data15', 'data16', 'data17', 'data18', 'data19', 'data20', 'data21', 'data22', 'data23', 'data24', 'data25', 'data26', 'data27', 'data28', 'data29', 'data30', 'data31', 'data32', 'data33', 'data34', 'data35', 'data36', 'data37', 'data38', 'data39', 'data40', 'data41', 'data42', 'data43', 'data44', 'data45', 'data46', 'data47', 'data48', 'data49', 'data50', 'data51', 'data52', 'data53', 'data54', 'data55', 'data56', 'data57', 'data58', 'data59', 'data60', 'data61', 'data62', 'data63', 'data64', 'data65', 'data66', 'data67', 'data68', 'data69', 'data70', 'data71', 'data72', 'data73', 'data74', 'data75', 'data76', 'data77', 'data78', 'data79', 'data80', 'data81', 'data82', 'data83', 'data84', 'data85', 'data86', 'data87', 'data88', 'data89', 'data90', 'data91', 'data92', 'data93', 'data94', 'data95', 'data96', 'data97', 'data98', 'data99', 'data100', 'data101', 'data102', 'data103', 'data104', 'data105', 'data106', 'data107', 'data108', 'data109', 'data110', 'data111', 'data112', 'data113', 'data114', 'data115', 'data116', 'data117', 'data118', 'data119', 'data120', 'data121', 'data122', 'data123', 'data124', 'data125', 'data126', 'data127', 'data128', 'data129', 'data130', 'data131', 'data132', 'data133', 'data134', 'data135', 'data136', 'data137', 'data138', 'data139', 'data140', 'data141', 'data142', 'data143', 'data144', 'data145', 'data146', 'data147', 'data148', 'data149', 'data150', 'data151', 'data152', 'data153', 'data154', 'data155', 'data156', 'data157', 'data158', 'data159', 'data160', 'data161', 'data162', 'data163', 'data164', 'data165', 'data166', 'data167', 'data168', 'data169', 'data170', 'data171', 'data172', 'data173', 'data174', 'data175', 'data176', 'data177', 'data178', 'data179', 'data180', 'data181', 'data182', 'data183', 'data184', 'data185', 'data186', 'data187', 'data188', 'data189', 'data190', 'data191', 'data192', 'data193', 'data194', 'data195', 'data196', 'data197', 'data198', 'data199', 'data200', 'data201', 'data202', 'data203', 'data204', 'data205', 'data206', 'data207', 'data208', 'data209', 'data210', 'data211', 'data212', 'data213', 'data214', 'data215', 'data216', 'data217', 'data218', 'data219', 'data220', 'data221', 'data222', 'data223', 'data224', 'data225', 'data226', 'data227', 'data228', 'data229', 'data230', 'data231', 'data232', 'data233', 'data234', 'data235', 'data236', 'data237', 'data238', 'data239', 'data240', 'data241', 'data242', 'data243', 'data244', 'data245', 'data246', 'data247', 'data248', 'data249', 'data250'};
    predictors = inputTable(:, predictorNames);
    response = inputTable.Brand;
    isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
    % Set up holdout validation
    cvp = cvpartition(response, 'Holdout', 0.25);
    trainingPredictors = predictors(cvp.training, :);
    trainingResponse = response(cvp.training, :);
    trainingIsCategoricalPredictor = isCategoricalPredictor;

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    template = templateSVM(...
        'KernelFunction', 'gaussian', ...
        'PolynomialOrder', [], ...
        'KernelScale', 16, ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    classificationSVM = fitcecoc(...
        trainingPredictors, ...
        trainingResponse, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', categorical({'Ford'; 'Honda'; 'Nissan'; 'Toyota'}));

    % Create the result struct with predict function
    svmPredictFcn = @(x) predict(classificationSVM, x);
    validationPredictFcn = @(x) svmPredictFcn(x);

    % Add additional fields to the result struct


    % Compute validation predictions
    validationPredictors = predictors(cvp.test, :);
    validationResponse = response(cvp.test, :);
    [validationPredictions, validationScores] = validationPredictFcn(validationPredictors);

    % Compute validation accuracy
    correctPredictions = (validationPredictions == validationResponse);
    isMissing = ismissing(validationResponse);
    correctPredictions = correctPredictions(~isMissing);
    validationAccuracy = sum(correctPredictions)/length(correctPredictions);
end

