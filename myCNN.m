% configuration
ei.imageDim = 160;  % image dimension
ei.numClasses = 2;  % number of class for classification, in this case, 2
ei.filterDims = [13,9];    % filter size for conv layer
ei.imageChannel = 1;
ei.numFilters = [30,10];    % number of conv kernel used in each conv layer
ei.poolDims = [2, 2];

% weight decay
ei.lambda = 0.0001;

% load image data
[images, labels, testImages, testLabels] = loadDCMImages(ei.imageDim);
images = reshape(images,ei.imageDim,ei.imageDim,1,[]);
testImages = reshape(testImages,ei.imageDim,ei.imageDim,1,[]);

% initialize weight
stack = initialize_weights(ei);

% configuration for minibatch SGD
options.epochs = 6;
opitons.minibatch = 100;
options.alpha = 1e-3;
options.momentum = .95;

epochs = options.epochs;
alpha = options.alpha;
minibatch = opitons.minibatch;
momentum = options.momentum;
m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;

velocity = cell(size(stack));

for l = 1:numel(stack)
    velocity{l}.W = zeros(size(stack{l}.W));
    velocity{l}.b = zeros(size(stack{l}.b));
end;

it = 0; % number of iteration
C = []; % cost of each iteration
A1 = [];    % accuracy of training set
A2 = [];    % accuracy of validation set
P = []; % predictions
t_train_0 = clock;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % momentum enable
        if it == momIncrease
            mom = momentum;
        end;

        % mini-batch pick
        mb_images = images(:,:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));
        
        % backpropogation
        [cost, acc, preds, grad_stack] = cnn_train(mb_images, mb_labels, stack, ei);      
        
        % renew the weight
        for l = 1:numel(stack)
            velocity{l}.W = mom*velocity{l}.W + alpha*(grad_stack{l}.W/minibatch+ei.lambda*stack{l}.W);
            velocity{l}.b = mom*velocity{l}.b + alpha*grad_stack{l}.b/minibatch;
            stack{l}.W = stack{l}.W - velocity{l}.W;
            stack{l}.b = stack{l}.b - velocity{l}.b;
        end;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;
        A1(length(A1)+1) = acc;
    end
    fprintf('epoch 1 train time is %f\n',etime(clock,t_train_0));
    alpha = alpha/2.0;
    
    t_test_0 = clock;
    
    activations1 = cnnConvolve4D(testImages, stack{1}.W, stack{1}.b);
    activationsPooled1 = cnnPool(ei.poolDims(1), activations1);
    activations2 = cnnConvolve4D(activationsPooled1, stack{2}.W, stack{2}.b);
    activationsPooled2 = cnnPool(ei.poolDims(2), activations2);

    % Reshape activations into 2-d matrix, hiddenSize x numImages,
    % for Softmax layer
    numTest = size(testImages,4);
    activationsPooled2 = reshape(activationsPooled2,[],numTest);


    probs = exp(bsxfun(@plus, stack{3}.W * activationsPooled2, stack{3}.b));
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);
    
    [~,preds] = max(probs,[],1);
    preds = preds';
    
    acc = sum(preds==testLabels)/length(preds);
    A2 = [A2, acc];
    P = [P preds];
    fprintf('epoch 1 test time is %f\n',etime(clock,t_test_0));
    fprintf('Accuracy is %f\n',acc);
end
save('preds4.mat','P');
save('train_accs4.mat','A1');
save('test_accs4.mat','A2');
save('cost4.mat','C');
save('probs4.mat', 'probs');




