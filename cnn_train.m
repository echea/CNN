function [ cost, acc, preds, grad_stack ] = cnn_train( images, labels, stack, ei )
% backpropogation algorithm
% Parameters:
%  images - image data
%  labels - labels for benign and malignant
%  stack - collection of weights for CNN
%  ei - configurations
%
% Returns:
%  cost - error
%  acc - accuracy
%  preds - predictions
%  grad_stack - collections of all gradient for weights
numImages = size(images,4);

grad_stack = cell(size(stack));
for i = 1:numel(stack)
    grad_stack{i}.W = zeros(size(stack{i}.W));
    grad_stack{i}.b = zeros(size(stack{i}.b));
end;

convDims = zeros(size(ei.filterDims));
outputDims = zeros(size(ei.filterDims));
convDims(1) = ei.imageDim - ei.filterDims(1) + 1;
outputDims(1) = (convDims(1)) / ei.poolDims(1);
convDims(2) = outputDims(1) - ei.filterDims(2) + 1;
outputDims(2) = (convDims(2))/ei.poolDims(2);

%Feedfoward Pass
activations1 = cnnConvolve(images, stack{1}.W, stack{1}.b);
activationsPooled1 = cnnPool(ei.poolDims(1), activations1);
activations2 = cnnConvolve(activationsPooled1, stack{2}.W, stack{2}.b);
activationsPooled2 = cnnPool(ei.poolDims(2), activations2);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled2 = reshape(activationsPooled2,[],numImages);

%% --------- Softmax Layer ---------

probs = exp(bsxfun(@plus, stack{3}.W * activationsPooled2, stack{3}.b));
sumProbs = sum(probs, 1);
probs = bsxfun(@times, probs, 1 ./ sumProbs);

[~,preds] = max(probs,[],1);
preds = preds';
acc = sum(preds==labels)/length(preds);

%% --------- Calculate Cost ----------
logp = log(probs);
index = sub2ind(size(logp),labels',1:size(probs,2));
ceCost = -sum(logp(index));        
wCost = ei.lambda/2 * (sum(stack{3}.W(:).^2)+sum(stack{1}.W(:).^2)+sum(stack{2}.W(:).^2));
cost = ceCost/numImages + wCost;

%% --------- Backpropagation ----------
%errors
% softmax layer
output = zeros(size(probs));
output(index) = 1;
DeltaSoftmax = (probs - output);
t = -DeltaSoftmax;

% error of second pooling layer
DeltaPool2 = reshape(stack{3}.W' * DeltaSoftmax,outputDims(2),outputDims(2),ei.numFilters(2),numImages);

DeltaUnpool2 = zeros(convDims(2),convDims(2),ei.numFilters(2),numImages);        
for imNum = 1:numImages
    for FilterNum = 1:ei.numFilters(2)
        unpool = DeltaPool2(:,:,FilterNum,imNum);
        DeltaUnpool2(:,:,FilterNum,imNum) = kron(unpool,ones(ei.poolDims(2)))./(ei.poolDims(2) ^ 2);
    end
end 

% error of second convolutional layer
DeltaConv2 = DeltaUnpool2 .* activations2 .* (1 - activations2);

%error of first pooling layer
DeltaPooled1 = zeros(outputDims(1),outputDims(1),ei.numFilters(1),numImages);
for i = 1:numImages
    for f1 = 1:ei.numFilters(1)
        for f2 = 1:ei.numFilters(2)
            DeltaPooled1(:,:,f1,i) = DeltaPooled1(:,:,f1,i) + convn(DeltaConv2(:,:,f2,i),stack{2}.W(:,:,f1,f2),'full');
        end
    end
end

%error of first convolutional layer
DeltaUnpool1 = zeros(convDims(1),convDims(1),ei.numFilters(1),numImages);
for imNum = 1:numImages
    for filterNum = 1:ei.numFilters(1)
        unpool = DeltaPooled1(:,:,filterNum,imNum);
        DeltaUnpool1(:,:,filterNum,imNum) = kron(unpool,ones(ei.poolDims(1)))./(ei.poolDims(1) ^ 2);
    end
end

DeltaConv1 = DeltaUnpool1 .* activations1 .* (1-activations1);

%% ---------- Gradient Calculation ----------
% softmax layer
grad_stack{3}.W = DeltaSoftmax*activationsPooled2';
grad_stack{3}.b = sum(DeltaSoftmax,2);

for fil2 = 1:ei.numFilters(2)
    for fil1 = 1:ei.numFilters(1)
        for im = 1:numImages
            grad_stack{2}.W(:,:,fil1,fil2) = grad_stack{2}.W(:,:,fil1,fil2) + conv2(activationsPooled1(:,:,fil1,im),rot90(DeltaConv2(:,:,fil2,im),2),'valid');
        end
    end
    temp = DeltaConv2(:,:,fil2,:);
    grad_stack{2}.b(fil2) = sum(temp(:));
end

% first convolutional layer
for fil1 = 1:ei.numFilters(1)
    for channel = 1:ei.imageChannel
        for im = 1:numImages
            grad_stack{1}.W(:,:,channel,fil1) = grad_stack{1}.W(:,:,channel,fil1) + conv2(images(:,:,channel,im),rot90(DeltaConv1(:,:,fil1,im),2),'valid');
        end
    end
    temp = DeltaConv1(:,:,fil1,:);
    grad_stack{1}.b(fil1) = sum(temp(:));
end

end

