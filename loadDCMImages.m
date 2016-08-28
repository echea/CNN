function [data_train, labels_train, data_test, labels_test] = loadDCMImages(input_dim)
list_b = dir([pwd,'\udata\0\']);
list_m = dir([pwd,'\udata\1\']);
addpath ../mycnn/udata/0/;
addpath ../mycnn/udata/1/;
N_b = length(list_b);
N_m = length(list_m);
b_index = (N_b-2)*5;
m_index = (N_m-2)*5;
data = zeros(input_dim, input_dim, b_index+m_index);
labels = zeros(b_index+m_index,1);
index = 1;
oindex = [];
for i = 3:N_b
    tmp = load(list_b(i).name);
    data(:,:,index) = tmp.patch;
    data(:,:,index+1) = imnoise(tmp.patch);
    data(:,:,index+2) = imnoise(tmp.patch);
    data(:,:,index+3) = imnoise(tmp.patch);
    data(:,:,index+4) = imnoise(tmp.patch);
    oindex = [oindex, index];
    index = index + 5;
end;

for i = 3:N_m
    tmp = load(list_m(i).name);
    data(:,:,index) = tmp.patch;
    data(:,:,index+1) = imnoise(tmp.patch);
    data(:,:,index+2) = imnoise(tmp.patch);
    data(:,:,index+3) = imnoise(tmp.patch);
    data(:,:,index+4) = imnoise(tmp.patch);
    labels(index:index+4) = 1;
    oindex = [oindex, index];
    index = index + 5;
end;

data = double(data)/255;
mean_data = mean(data,3);
for i = 1:index-1
    data(:,:,i) = data(:,:,i) - mean_data;
end;
labels = labels+1;
rand('seed',10);
data_train = data(:,:,[1:ceil(b_index*3/4), b_index+1: b_index+ceil(m_index*3/4)]);
labels_train = labels([1:ceil(b_index*3/4), b_index+1: b_index+ceil(m_index*3/4)]);
rand_index = randperm(size(data_train,3));
data_train = data_train(:,:,rand_index);
labels_train = labels_train(rand_index);
data_test = data(:,:,oindex([ceil(b_index*3/4/5)+1:b_index/5, b_index/5+ceil(m_index*3/4/5)+1:end]));
labels_test = labels(oindex([ceil(b_index*3/4/5)+1:b_index/5, b_index/5+ceil(m_index*3/4/5)+1:end]));
end