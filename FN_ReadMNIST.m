%ReadMNIST
%Read MNIST Database
if (~exist('indir','var'))
 indir = pwd;
end
fid1 = fopen('t10k-labels.idx1-ubyte','r','ieee-be');
test_labels = fread(fid1,'uint8');
fid2 = fopen('train-labels.idx1-ubyte','r','ieee-be');
train_labels = fread(fid2,'uint8');
fid3 = fopen('t10k-images.idx3-ubyte','r','ieee-be');
test_images = fread(fid3,'uint8');
fid4 = fopen('train-images.idx3-ubyte','r','ieee-be');
train_images = fread(fid4,'uint8');
train_labels_idx=9;
train_images_idx=17;
test_labels_idx=9;
test_images_idx=17;

%train_images=double(train_images>0);
%test_images=double(test_images>0);

%}