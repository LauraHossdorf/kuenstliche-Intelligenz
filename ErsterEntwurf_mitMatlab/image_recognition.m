clear all;
tic

imds = imageDatastore('C:\Users\Maria\Desktop\ki_matlab\imageRecognition', ...
'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

numTrainFiles = 100;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


layers = [
    imageInputLayer([200 200 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs',15, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',100, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(imdsTrain,layers,options);


YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

% cd C:\Users\Maria\Desktop\ki_matlab\test2
% 
% test = imread('test028003.png');
% test2 = uint8(255 * test);
% human1 = classify(net,test2)
% 
% test = imread('test045009.png');
% test2 = uint8(255 * test);
% human2 = classify(net,test2)
% 
% test = imread('test047011.png');
% test2 = uint8(255 * test);
% human3 = classify(net,test2)
% 
% test = imread('test051005.png');
% test2 = uint8(255 * test);
% animal1 = classify(net,test2)
% 
% test = imread('test052024.png');
% test2 = uint8(255 * test);
% human4 = classify(net,test2)
% 
% test = imread('test059002.png');
% test2 = uint8(255 * test);
% animal2 = classify(net,test2)
% 
% test = imread('test062001.png');
% test2 = uint8(255 * test);
% animal3 = classify(net,test2)
% 
% test = imread('test075002.png');
% test2 = uint8(255 * test);
% animal4 = classify(net,test2)
% 
% test = imread('test106001.png');
% test2 = uint8(255 * test);
% animal5 = classify(net,test2)
% 
% test = imread('test129002.png');
% test2 = uint8(255 * test);
% human5 = classify(net,test2)


toc
