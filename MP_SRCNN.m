function im_h = MP_SRCNN(model, im_b)

%% load CNN model parameters
load(model);
[conv1_patchsize2,conv1_filters] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
[conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
[hei, wid] = size(im_b);


%% channel 1
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data1 = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data1(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data1(:,:,i) = max(conv1_data1(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data1 = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data1(:,:,i) = conv2_data1(:,:,i) + imfilter(conv1_data1(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data1(:,:,i) = max(conv2_data1(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data1 = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data1(:,:) = conv3_data1(:,:) + imfilter(conv2_data1(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% channel 2
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data2 = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data2(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data2(:,:,i) = max(conv1_data2(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data2 = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data2(:,:,i) = conv2_data2(:,:,i) + imfilter(conv1_data2(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data2(:,:,i) = max(conv2_data2(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data2 = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data2(:,:) = conv3_data2(:,:) + imfilter(conv2_data2(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% channel 3
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data3 = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data3(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data3(:,:,i) = max(conv1_data3(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data3 = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data3(:,:,i) = conv2_data3(:,:,i) + imfilter(conv1_data3(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data3(:,:,i) = max(conv2_data3(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data3 = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data3(:,:) = conv3_data3(:,:) + imfilter(conv2_data3(:,:,i), conv3_subfilter, 'same', 'replicate');
end


%% channel 4
%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data4 = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data4(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data4(:,:,i) = max(conv1_data4(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data4 = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data4(:,:,i) = conv2_data4(:,:,i) + imfilter(conv1_data4(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data4(:,:,i) = max(conv2_data4(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data4 = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data4(:,:) = conv3_data4(:,:) + imfilter(conv2_data4(:,:,i), conv3_subfilter, 'same', 'replicate');
end





%% SRCNN reconstruction
im_h = (conv3_data1(:,:) +conv3_data2(:,:)+conv3_data3(:,:)+conv3_data4(:,:))/4+ biases_conv3*1.4;