% Ref: https://github.com/hagaygarty/mdCNN/blob/master/Demo/MNIST3d/getMNIST3Ddata.m
function [res] = Transform2dto3d(image,len)
    res = zeros([size(image) size(image,1)], 'uint8');
    for z=0:(len-1)
        SE = strel('diamond',len-z);
        deIm = imdilate(image,SE,'same');
        edgeIm = uint8(255*edge(deIm));
        res(:,:, round(size(res,3)/2)+z) = edgeIm;
        res(:,:, round(size(res,3)/2)-z) = edgeIm;
    end
    z=len;
    res(:,:, round(size(res,3)/2)+z) = image;
    res(:,:, round(size(res,3)/2)-z) = image;
end
