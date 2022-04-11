import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from main import SAR_BagNet
import torch
import cv2
import seaborn as sns
import os
from matplotlib.pyplot import savefig
def preprocess_image(img, use_cuda=1):
    image = img/255
    means = (0.0293905372581, 0.0293905372581, 0.0293905372581)
    stds = (0.0308426998737, 0.0308426998737, 0.0308426998737)
    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return preprocessed_img_tensor
def feature_heatmap(model,image,resize_shape,output_file):
    model.eval()
    image = cv2.resize(image, resize_shape)
    image = preprocess_image(image, use_cuda=0)
    image = image.type(torch.FloatTensor)
    image = image.cuda()
    #predict=model(image)
    predict,featuremap = model(image)
    heatmap=featuremap
    prob = F.softmax(predict, dim=1).data.squeeze()
    # print(predict)
    # print(prob)
    max=np.max(heatmap)
    plt.cla()
    sns.set()
    ax = sns.heatmap(heatmap,  fmt='.1f',cmap=plt.cm.jet,vmin=-max,vmax=max)
    savefig(output_file)
    #plt.show()
    plt.close('all')
    return featuremap
if __name__ == '__main__':
    model= SAR_BagNet.BagNet18(pretrained=True).cuda()
    input_path='image_file'
    output_path = 'output_file'
    files = os.listdir(input_path)
    for imgname in files:
        if imgname.endswith('jpg'):
            input_img = input_path + imgname
            print('imgname:', imgname)
            img_label = 0
            image=cv2.imread(input_img)
            output_file = output_path + imgname[:-4] +'new'+ 'heatmap.JPG'
            heatmap = feature_heatmap(model, image, (100, 100),output_file)
    heatmap=feature_heatmap(model,image,(100,100))