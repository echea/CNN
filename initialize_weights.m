function [ stack ] = initialize_weights( ei )
% initalize weights for all layers
% Parameters:
%  ei - configurations
%
% Returns:
%  stack - collections of all weights
stack = cell(1, numel(ei.filterDims)+1);
stack{1}.W = 1e-1*randn(ei.filterDims(1),ei.filterDims(1),ei.imageChannel,ei.numFilters(1));
stack{2}.W = 1e-1*randn(ei.filterDims(2),ei.filterDims(2),ei.numFilters(1),ei.numFilters(2));
stack{1}.b = zeros(ei.numFilters(1), 1);
stack{2}.b = zeros(ei.numFilters(2), 1);
stack{3}.b = zeros(ei.numClasses, 1);
outDim1 = ei.imageDim - ei.filterDims(1) + 1; % dimension of convolved image
outDim1 = outDim1/ei.poolDims(1);
outDim2 = outDim1 - ei.filterDims(2) + 1; % dimension of convolved image
outDim2 = outDim2/ei.poolDims(2);
hiddenSize = outDim2^2*ei.numFilters(2);

r  = sqrt(6) / sqrt(ei.numClasses+hiddenSize+1);
stack{3}.W = rand(ei.numClasses, hiddenSize) * 2 * r - r;

end

