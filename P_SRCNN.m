function im_h = P_SRCNN(model, im_b)

%% load CNN model parameters
load(model);
[conv1_patchsize2,conv1_filters] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
[conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
[hei, wid] = size(im_b);


%% CNN 1
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data1p = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data1p(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data1p(:,:,i) = max(conv1_data1p(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data1p = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data1p(:,:,i) = conv2_data1p(:,:,i) + imfilter(conv1_data1p(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data1p(:,:,i) = max(conv2_data1p(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data1p = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data1p(:,:) = conv3_data1p(:,:) + imfilter(conv2_data1p(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% CNN 2
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data2p = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data2p(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data2p(:,:,i) = max(conv1_data2p(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data2p = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data2p(:,:,i) = conv2_data2p(:,:,i) + imfilter(conv1_data2p(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data2p(:,:,i) = max(conv2_data2p(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data2p = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data2p(:,:) = conv3_data2p(:,:) + imfilter(conv2_data2p(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% CNN 3
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data3 = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data3(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data3(:,:,i) = max(conv1_data3(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data3p = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data3p(:,:,i) = conv2_data3p(:,:,i) + imfilter(conv1_data3(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data3p(:,:,i) = max(conv2_data3p(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data3p = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data3p(:,:) = conv3_data3p(:,:) + imfilter(conv2_data3p(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% CNN 4
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data4p = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data4p(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data4p(:,:,i) = max(conv1_data4p(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data4p = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data4p(:,:,i) = conv2_data4p(:,:,i) + imfilter(conv1_data4p(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data4p(:,:,i) = max(conv2_data4p(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data4p = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data4p(:,:) = conv3_data4p(:,:) + imfilter(conv2_data4p(:,:,i), conv3_subfilter, 'same', 'replicate');
end





%% SRCNN reconstruction
im_h = (conv3_data1p(:,:) +conv3_data2p(:,:)+conv3_data3p(:,:)+conv3_data4p(:,:))/4+ biases_conv3*1.6;