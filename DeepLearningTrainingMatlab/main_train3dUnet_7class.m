% get directory name where the training data is found
imageDir = fullfile('C:\Users\TEV4\DLtraining\','VesTS_3Dmulticlass');

%% Create data stores for Training data
% create an image datastore
volReader = @(x) matRead(x);
volLoc = fullfile(imageDir,'imagesTr');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);

% create a pixelLabel datastore for Training data
lblLoc = fullfile(imageDir,'labelsTr');
classNames = ["background","BM","Lumen","Nuclei","Mito","ER","Unknown"];
pixelLabelID = [0 1 2 3 4 5 6];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);

% create random patch datastore for Training data
patchSize = [164 164 100];
patchPerImage = 60;
miniBatchSize = 12; 
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

%% Create datastores for Validation data
% repeat creating datastores for the validation data
volLocVal = fullfile(imageDir,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(imageDir,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;

%% Create Unet Layers
numChannels = 1;
inputPatchSize = [patchSize numChannels];
numClasses = 7;
[lgraph,outPatchSize] = unet3dLayers(inputPatchSize,numClasses,'ConvolutionPadding','valid');

%% data augmentation
dataSource = 'Training';
dsTrain = transform(patchds,@(patchIn)augmentAndCrop3dPatch(patchIn,outPatchSize,dataSource));

dataSource = 'Validation';
dsVal = transform(dsVal,@(patchIn)augmentAndCrop3dPatch(patchIn,outPatchSize,dataSource));


inputLayer = image3dInputLayer(inputPatchSize,'Normalization','none','Name','ImageInputLayer');
lgraph = replaceLayer(lgraph,'ImageInputLayer',inputLayer);


%% training options
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'InitialLearnRate',5e-4, ... 
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);

%% train network
 modelDateTime = string(datetime('now','Format',"yyyy-MM-dd-HH-mm-ss"));
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(strcat("trained3DUNet-",modelDateTime,"-Epoch-",num2str(options.MaxEpochs),".mat"),'net');