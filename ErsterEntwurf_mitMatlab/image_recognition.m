clear all;
tic

imds = imageDatastore('C:\Users\Maria\Desktop\ki_matlab\imageRecognition', ... %Speicherort der Bilder
'IncludeSubfolders',true,'LabelSource','foldernames'); 
labelCount = countEachLabel(imds) %labels= human/animal. Zählt wie viele Bilder die Labels haben

numTrainFiles = 100; %nimmt 100 Bilder aus jedem Ordner, wendet Filter an und versucht ALLEINE Gemeinsamkeiten und Unterschiede zu finden
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); %Validation -> die restlichen Bilder sollen kategorisiert werden.

layers = [ 
    imageInputLayer([200 200 1]) %1, weil einfarbig
    
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
    
    fullyConnectedLayer(2) %2 Labels
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ... 
    'MaxEpochs',10, ... %wie oft das "Training" wiederholt werden soll. Bei mehr Epochs hat sich das Programm nicht verbessert, wegen overfitting
    'ValidationData',imdsValidation, ... %welche Daten es zur Validation benutzen soll
    'ValidationFrequency',100, ... %wie oft validiert werden soll
    'Verbose',false, ... 
    'Plots','training-progress' ); %Diagramm

programm = trainNetwork(imdsTrain,layers,options); %Befehl: nimm die gegebenen Bilder, Layer und Optionen und führ das Training


% cd C:\Users\Maria\Desktop\ki_matlab\test2
% test = imread('');
% test2 = uint8(test);
% label = classify(programm,test2)


toc

%Verbesserung: 1 mehr Bilder fürs Training 2 bessere Bilder 3 Layer verändern
