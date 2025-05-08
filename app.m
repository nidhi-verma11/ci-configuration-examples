%% Step 1: Load the CSV File
%csvFile = 'C:\Users\Nidhi Verma\Downloads\oasis_cross-sectional.csv';
csvFile = "C:\Users\Nidhi Verma\Desktop\Alzheimer detection project\oasis_cross-sectional.csv"; 
csvData = readtable(csvFile, 'VariableNamingRule', 'preserve');
%disp('CSV file loaded successfully.');
%disp('Preview of the CSV file:');
%disp(head(csvData));
%% Step 2: Handle Missing Data
missingRows = any(ismissing(csvData), 2);
if any(missingRows)
 %fprintf('Number of rows with missing values: %d\n', sum(missingRows));
 csvData(missingRows, :) = []; 
 %disp('Rows with missing data have been removed.');
else
 %disp('No missing data found in the CSV.');
end
%% Step 3: Remove Duplicate Rows
if height(csvData) ~= height(unique(csvData, 'rows'))
 %disp('Duplicate rows found.');
 csvData = unique(csvData, 'rows');
 %disp('Duplicate rows have been removed.');
else
 %disp('No duplicate rows found in the CSV.');
end
%% Step 4: Validate and Preprocess Images
imageDir = "C:\Users\Nidhi Verma\Desktop\Alzheimer detection project\MRI IMAGING\MRI Data"; 
classFolders = dir(imageDir);
targetSize = [224, 224];
processedImages = [];
matchedRows = [];
imageIDs = {};
%disp('Starting image validation and preprocessing...');
for i = 1:height(csvData)
 imageID = csvData.ID{i}; 
 classLabel = ''; 
 
 for folder = 3:length(classFolders) 
 folderName = classFolders(folder).name;
 folderPath = fullfile(imageDir, folderName);
 imagePath = fullfile(folderPath, [imageID, '_mpr-1_100.jpg']); 
 
 if exist(imagePath, 'file')
 img = imread(imagePath);
 resizedImg = imresize(img, targetSize);
 grayImg = rgb2gray(resizedImg); 
 denoisedImg = medfilt2(grayImg); 
 
 processedImages = cat(4, processedImages, denoisedImg); 
 matchedRows = [matchedRows; csvData(i, :)]; 
 imageIDs{end+1} = imageID; 
 classLabel = folderName; 
 break; 
 end
 end
 
 if isempty(classLabel)
 %warning('No matching image found for ID: %s', imageID);
 end
end
if isempty(processedImages)
 %error('No images matched the CSV data. Please check the file names and directory structure.');
end
%disp('Image validation and preprocessing completed.');
numProcessedImages = size(processedImages, 4);
%disp(['Number of images processed: ', num2str(numProcessedImages)]);
%disp(['Number of training samples: ', num2str(height(csvData))]); 
%% Step 5: Save Cleaned Data
cleanedCSVData = matchedRows; 
save('cleanedData.mat', 'cleanedCSVData', 'processedImages');
%disp('Cleaned data saved to "cleanedData.mat".');
%% Step 6: Validate Data Accuracy
cleanedData = cleanedCSVData; 
%disp('Validating data accuracy...');
%% Step 6.1: Check for Consistency
% Check for missing values
missingValues = sum(ismissing(cleanedData));
%disp('Missing Values in Each Column:');
%disp(missingValues);
% Check for duplicates
[~, uniqueIdx] = unique(cleanedData, 'rows', 'stable'); 
duplicateRowsCount = height(cleanedData) - length(uniqueIdx); % Count duplicates
%disp('Number of Duplicate Rows:');
%disp(duplicateRowsCount);
if duplicateRowsCount > 0
 %disp('Duplicate rows found.');
else
 %disp('No duplicate rows found in the CSV.');
end
% Step 6.2: Statistical Analysis
% Summary statistics
summaryStats = summary(cleanedData);
%disp('Summary Statistics of Cleaned Data:');
%disp(summaryStats);
% Check for outliers using manual z-score calculation for numeric columns
numericData = cleanedData{:, varfun(@isnumeric, cleanedData, 'OutputFormat', 'uniform')}; % Extract numeric columns
% Calculate mean and standard deviation
meanValues = mean(numericData, 'omitnan'); % Omit NaN values
stdValues = std(numericData, 'omitnan'); % Omit NaN values
% Calculate z-scores manually
zScores = (numericData - meanValues) ./ stdValues;
% Identify outliers (z-score > 3 or < -3)
outliers = abs(zScores) > 3; % Create a logical array for outliers
%disp('Outliers Detected (True indicates an outlier):');
%disp(outliers);
%% Step 7: Exploratory Data Analysis (EDA)
%disp('Performing Exploratory Data Analysis (EDA)...');
% Step 7.1: Summary Statistics for Numeric Data
%disp('Summary Statistics of Cleaned Data:');
summaryStats = summary(cleanedData);
%disp(summaryStats);
% Step 7.2: Visualize Distributions of Numeric Variables
numericColumns = cleanedData{:, varfun(@isnumeric, cleanedData, 'OutputFormat', 'uniform')}; % Extract numeric columns
numericVarNames = cleanedData.Properties.VariableNames(varfun(@isnumeric, cleanedData, 'OutputFormat', 'uniform'));
% Create histograms for each numeric variable using hist
% Step 7.3: Manual Box Plots for Outlier Detection
%figure;
for i = 1:length(numericVarNames)
 %subplot(ceil(length(numericVarNames)/2), 2, i); % Arrange in a grid
 % Calculate quartiles and outliers
 q1 = prctile(numericColumns(:, i), 25);
 q2 = prctile(numericColumns(:, i), 50);
 q3 = prctile(numericColumns(:, i), 75);
 iqr = q3 - q1; % Interquartile range
 lowerBound = q1 - 1.5 * iqr; % Lower bound for outliers
 upperBound = q3 + 1.5 * iqr; % Upper bound for outliers
 
 % Create box plot manually
 %hold on;
 % Draw the box
 %fill([1 1 2 2], [q1 q3 q3 q1], 'b', 'FaceAlpha', 0.5);
 % Draw the median line
 %plot([1 2], [q2 q2], 'k', 'LineWidth', 2);
 % Draw the whiskers
 %plot([1 1], [min(numericColumns(numericColumns(:, i) >= lowerBound, i)) q1], 'k');
 %plot([2 2], [q3 max(numericColumns(numericColumns(:, i) <= upperBound, i))], 'k');
 
 % Plot outliers
 outliers = numericColumns(numericColumns(:, i) < lowerBound | numericColumns(:, i) > upperBound, i);
 %plot(1 + rand(size(outliers)) * 0.1, outliers, 'ro'); % Random jitter for visibility
 
 %title(['Manual Box Plot of ', numericVarNames{i}]);
 %xlim([0.5 2.5]);
 %xticks([1 2]);
 %xticklabels({numericVarNames{i}, ''});
 %ylabel('Values');
 %hold off;
end
%sgtitle('Manual Box Plots of Numeric Variables');
% Step 7.4: Manual Calculation of Correlation Matrix
numVars = size(numericColumns, 2); % Number of numeric variables
covarianceMatrix = zeros(numVars); % Initialize covariance matrix
for i = 1:numVars
 for j = 1:numVars
 % Calculate covariance
 x = numericColumns(:, i);
 y = numericColumns(:, j);
 
 % Remove NaN values
 validIdx = ~isnan(x) & ~isnan(y); % Indices of valid (non-NaN) entries
 xValid = x(validIdx);
 yValid = y(validIdx);
 
 % Calculate means
 meanX = mean(xValid);
 meanY = mean(yValid);
 
 % Calculate covariance
 covarianceMatrix(i, j) = mean((xValid - meanX) .* (yValid - meanY));
 end
end
% Display the covariance matrix
%disp('Covariance Matrix of Numeric Variables:');
%disp(covarianceMatrix);
%% Step 8: EDA for Image Dataset
%disp('Performing EDA for Image Dataset...');
% Step 8.1: Basic Image Statistics
numImages = size(processedImages, 4); % Number of images
%disp(['Number of images processed: ', num2str(numImages)]);
% Get dimensions of the first image
if numImages > 0
 [imgHeight, imgWidth, imgChannels] = size(processedImages(:, :, :, 1));
 %disp(['Image dimensions: ', num2str(imgHeight), ' x ', num2str(imgWidth), ' x ', num2str(imgChannels)]);
end
% Step 8.2: Visualize Sample Images
%figure;
for i = 1:min(9, numImages) 
 %subplot(3, 3, i);
 
 % Extract the i-th image from the 4D array
 img = processedImages(:, :, :, i);
 
 % Check if the image is grayscale or RGB
 if size(img, 3) == 1
 % Grayscale image
 %imshow(img, []);
 else
 % RGB image
 %imshow(img);
 end
 
% title(['Sample Image ', num2str(i)]);
end
%sgtitle('Sample Images from Dataset');
% Step 8.3: Image Quality Assessment (e.g., Mean Brightness)
meanBrightness = zeros(numImages, 1);
for i = 1:numImages
 meanBrightness(i) = mean(processedImages(:, :, :, i), 'all'); % Calculate mean brightness
end
%disp(['Mean Brightness of Images: ', num2str(mean(meanBrightness))]);
% Step 8.4: Distribution of Class Labels
if isfield(cleanedCSVData, 'ClassLabel') 
 classCounts = countcats(categorical(cleanedCSVData.ClassLabel));
 
end
%disp('Exploratory Data Analysis (EDA) completed.');
% Display the variable names in cleanedData
%disp('Variable names in cleanedData:');
%disp(cleanedData.Properties.VariableNames);
%% Step 9: Feature Selection
%disp('Performing Feature Selection...');
% Step 9.1: Initialize selectedFeatures
selectedFeatures = {}; % Initialize as an empty cell array
% Check variable names in cleanedData
%disp('Variable names in cleanedData:');
%disp(cleanedData.Properties.VariableNames);
% Set the target variable to 'CDR'
targetVariable = 'CDR'; 
if ismember(targetVariable, cleanedData.Properties.VariableNames)
 % Calculate correlation with the target variable
 numericColumns = cleanedData{:, varfun(@isnumeric, cleanedData, 'OutputFormat', 'uniform')}; % Extract numeric columns
 targetData = cleanedData{:, targetVariable}; % Extract target variable
 correlationWithTarget = corr(numericColumns, targetData, 'Rows', 'complete'); % Calculate correlation
% disp('Correlation with Target Variable:');
 %disp(correlationWithTarget);
 
 % Select features with correlation above a threshold (e.g., 0.3)
 threshold = 0.3;
 selectedFeatures = cleanedData.Properties.VariableNames(abs(correlationWithTarget) > threshold);
 %disp('Selected Features based on Correlation:');
 %disp(selectedFeatures);
else
 %disp('Target variable not found in cleaned data.');
end
% Step 9.2: Variance Threshold
varianceThreshold = 0.01; 
variances = var(numericColumns, 0, 1, 'omitnan'); % Calculate variance for each feature
%disp('Feature Variances:');
%disp(variances);
% Select features with variance above the defined threshold
highVarianceFeatures = cleanedData.Properties.VariableNames(variances > varianceThreshold);

%disp('Selected Features based on Variance:');
%disp(highVarianceFeatures);
% Ensure both selectedFeatures and highVarianceFeatures are cell arrays
if ischar(selectedFeatures) || isstring(selectedFeatures)
 selectedFeatures = cellstr(selectedFeatures);
end
if ischar(highVarianceFeatures) || isstring(highVarianceFeatures)
 highVarianceFeatures = cellstr(highVarianceFeatures); 
end
% Combine selected features from both methods
finalSelectedFeatures = unique([selectedFeatures(:); highVarianceFeatures(:)]); 
%disp('Final Selected Features:');
%disp(finalSelectedFeatures);
%disp('Feature Selection completed.');
%% Step 10: Data Splitting
% check finalSelectedFeatures contains only numeric features
numericFeatureNames = cleanedData.Properties.VariableNames(varfun(@isnumeric, cleanedData, 'OutputFormat', 'uniform'));
finalSelectedFeatures = intersect(finalSelectedFeatures, numericFeatureNames); % Keep only numeric features
% Define X using the selected features
X = cleanedData{:, finalSelectedFeatures}; % Features
% Check if the target variable exists
if ismember(targetVariable, cleanedData.Properties.VariableNames)
 y = cleanedData{:, targetVariable}; % Target variable
else
 error('Target variable not found in cleaned data.');
end
% Split the data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
idx = cv.test;
% Separate to training and test data
XTrain = X(~idx, :);
yTrain = y(~idx, :);
XTest = X(idx, :);
yTest = y(idx, :);
%disp('Data splitting completed.');
% Assuming the target variable is in the cleaned CSV data
if ismember(targetVariable, cleanedData.Properties.VariableNames)
 y = cleanedData{:, targetVariable}; % Extract target variable
 yTrain = y(~idx); % Training labels
 yTest = y(idx); % Test labels
else
 %error('Target variable not found in cleaned data.');
end
%% Step 11: Data Normalization
% Normalize the training data
XTrainNorm = (XTrain - mean(XTrain)) ./ std(XTrain);
% Normalize the test data using the training data statistics
XTestNorm = (XTest - mean(XTrain)) ./ std(XTrain);
%disp('Data normalization completed.');
%% Step 12: Feature Extraction from Images
%disp('Extracting features from images using a pre-trained CNN...');
% Load a pre-trained CNN model (e.g., ResNet-50)
net = resnet50; % You can choose any other model as well
% Initialize feature extraction
numImages = size(processedImages, 4); % Number of images
cnnFeatures = []; % Initialize an array to store CNN features
% Loop through each image to extract features
for i = 1:numImages
 img = processedImages(:, :, :, i);
 
 % Resize the image to the input size of the CNN
 imgResized = imresize(img, [224, 224]); % Adjust size according to the model
 
 % Convert to RGB if the image is grayscale
 if size(imgResized, 3) == 1
 imgResized = repmat(imgResized, [1, 1, 3]); % Replicate the single channel
 end
 
 imgResized = im2double(imgResized); % Convert to double for CNN input
 
 % Extract features using the CNN
 features = activations(net, imgResized, 'fc1000', 'OutputAs', 'rows'); % 'fc1000' is the last fully connected layer
 cnnFeatures = [cnnFeatures; features]; % Append features for each image
end
%disp('Local feature extraction from images completed.');
%% Step 13: Combine Features (Fixed)
%disp('Combining extracted features from structured data and images...');
% Convert imageIDs and trainingIDs to strings (Ensure consistent data type)
imageIDs = string(imageIDs);
trainingIDs = string(cleanedCSVData.ID(~idx)); % Extract IDs for training samples
% Debug: Print sample IDs
%disp('Sample Training IDs:');
%disp(trainingIDs(1:min(5, length(trainingIDs)))); % Display first few IDs
%disp('Sample Image IDs:');
%disp(imageIDs(1:min(5, length(imageIDs)))); % Display first few IDs
% Identify valid indices where imageIDs match trainingIDs
[isMatch, matchIdx] = ismember(trainingIDs, imageIDs);
% Debug: Check how many matches were found
%disp(['Total Matches Found: ', num2str(sum(isMatch))]);
% Ensure `cnnFeaturesFiltered` only contains features for matched IDs
cnnFeaturesFiltered = cnnFeatures(matchIdx(isMatch), :);
% Final check before concatenation
if size(cnnFeaturesFiltered, 1) ~= sum(isMatch)
 error(['Row mismatch still exists after filtering. Found: ', ...
 num2str(size(cnnFeaturesFiltered, 1)), ' rows in cnnFeaturesFiltered vs ', ...
 num2str(sum(isMatch)), ' in structured data.']);
end
% Filter the structured data to match the valid indices
XTrainFiltered = XTrainNorm(isMatch, :);
yTrainFiltered = yTrain(isMatch);
% Ensure that the number of observations matches after filtering
if size(XTrainFiltered, 1) ~= size(cnnFeaturesFiltered, 1)
 %error('The number of observations in XTrainFiltered does not match the number of observations in cnnFeaturesFiltered.');
end
% Concatenate structured and image-based features
XTrainFinal = [XTrainFiltered, cnnFeaturesFiltered];
% Repeat for Test Data
testIDs = string(cleanedCSVData.ID(idx));
[isMatchTest, matchIdxTest] = ismember(testIDs, imageIDs);
cnnFeaturesTestFiltered = cnnFeatures(matchIdxTest(isMatchTest), :);
if size(cnnFeaturesTestFiltered, 1) ~= sum(isMatchTest)
 error(['Row mismatch in test data after filtering. Found: ', ...
 num2str(size(cnnFeaturesTestFiltered, 1)), ' rows in cnnFeaturesTestFiltered vs ', ...
 num2str(sum(isMatchTest)), ' in XTestNorm.']);
end
% Filter the test data to match the valid indices
XTestFiltered = XTestNorm(isMatchTest, :);
% Ensure that the number of observations matches after filtering
if size(XTestFiltered, 1) ~= size(cnnFeaturesTestFiltered, 1)
 %error('The number of observations in XTestFiltered does not match the number of observations in cnnFeaturesTestFiltered.');
end
% Final concatenation for test data
XTestFinal = [XTestFiltered, cnnFeaturesTestFiltered];
%disp('Feature combination completed successfully.');
% Example training data (replace with your actual data)
X_train = rand(100, 128); % 100 samples, 128 features
y_train = randi([0, 1], 100, 1); % Binary target variable
% Train and save the model
trainAndSaveModel(X_train, y_train);
% Call the UI function
dementia_detection_app(); % Ensure this is called after all other code
% Function definitions should be at the end of the script or in separate files
function dementia_detection_app
 % Create UI figure
 fig = uifigure('Name', 'Dementia Detection App', 'Position', [100 100 400 300]);
 
 % Upload Button
 btnUpload = uibutton(fig, 'push', 'Text', 'Upload Image', ...
 'Position', [150 230 100 30], ...
 'ButtonPushedFcn', @(btn, event) uploadImage(fig));
end
% Function to upload image
function uploadImage(fig)
 [file, path] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files'});
 if isequal(file, 0)
 uialert(fig, 'No file selected', 'Error');
 else
 % Remove existing axes and buttons
 delete(findall(fig, 'Type', 'axes'));
 delete(findall(fig, 'Type', 'uibutton', 'Text', 'Predict'));
 
 % Read and display the image
 img = imread(fullfile(path, file));
 ax = uiaxes(fig, 'Position', [50 50 300 150]);
 imshow(img, 'Parent', ax);
 drawnow; % UI update
 
 % Predict Button
 btnPredict = uibutton(fig, 'push', 'Text', 'Predict', ...
 'Position', [150 10 100 30], ...
 'ButtonPushedFcn', @(btn, event) predictDementia(img, fig));
 end
end
% Function to preprocess image
function processed_image = preprocess_image(img)
 % Convert to grayscale if needed
 if size(img, 3) == 3
 img = rgb2gray(img);
 end
 
 % Resize to 128x128 and normalize
 img = imresize(img, [128, 128]);
 processed_image = double(img) / 255;
end
% Function to predict dementia
function predictDementia(img, fig)
 processed_image = preprocess_image(img);
 result = predict_dementia(processed_image);
 
 if result == 1
 msg = 'Dementia Detected';
 else
 msg = 'No Dementia Detected';
 end
 
 % Show result in alert box
 uialert(fig, msg, 'Result');
end
% Function to predict dementia using SVM
function result = predict_dementia(image)
 % CNN feature extraction (Example for random data)
 cnn_features = rand(1, 128); % Replace with actual feature extraction logic
 % Load SVM model
 if exist('svm_model.mat', 'file') == 2
 loadedData = load('svm_model.mat'); % Load model
 model = loadedData.model_variable; % Extract the model from the loaded data
 else
 error('File svm_model.mat not found in the current directory.');
 end
 % Make predictions
 prediction = predict(model, cnn_features); % Use the appropriate prediction function
 % Return the prediction result
 result = prediction; % Adjust based on your model's output
end
% Function to train and save the SVM model (Call this once)
function trainAndSaveModel(X_train, y_train)
 % Train the SVM model
 model_variable = fitcsvm(X_train, y_train); % Replace with your actual training function
 % Save the trained model to a .mat file
 save('svm_model.mat', 'model_variable');
 
 %disp('Model saved as svm_model.mat');
end
