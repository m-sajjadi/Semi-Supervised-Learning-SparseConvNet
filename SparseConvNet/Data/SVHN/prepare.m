clc
clear all

load train_32x32.mat

xtrain = zeros(32,32,length(y),'uint8');
for i=1:length(y)
    if mod(i,1000) == 0
        fprintf('processed %d images\n',i);
    end
    xtrain(:,:,i) = rgb2gray(X(:,:,:,i))';
end

xtrain = reshape(xtrain,32*32,length(y));
ytrain = uint8(y'-1);

% % % 
load test_32x32.mat

xtest = zeros(32,32,length(y),'uint8');
for i=1:length(y)
    if mod(i,1000) == 0
        fprintf('processed %d images\n',i);
    end
    xtest(:,:,i) = rgb2gray(X(:,:,:,i))';
end

xtest = reshape(xtest,32*32,length(y));
ytest = uint8(y'-1);

% % % 

train = [ytrain; xtrain];
test = [ytest; xtest];

fileID = fopen('train.bin','w');
fwrite(fileID,train);
fclose(fileID);

fileID = fopen('test.bin','w');
fwrite(fileID,test);
fclose(fileID);
