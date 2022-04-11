# SAR-BagNet with PyTorch

PyTorch implementation of SAR-BagNet

## Requirements
   1. Pytorch platform for Windows 

   2. python 3.6+     

   3. The training model requires a video card with more than 12G video memory
    
   4. opencv
##Train model
1. Unzip the image. we provide the MSTAR dataset in the images folder 

2. run trian_test.py.The training process is the same as a traditional CNNs.  This program includes the preprocessing operation of the data of this project, and different processing processes can be selected according to different tasks

3. utils.py can generate heatmaps of each SAR images
##Generate heatmap for SAR images
   1. Please place a trained model in the specified folder,Model_urls is the location of the model, and model_dir is the save folder for the model,for example:
```
model_urls = {'SAR_BagNet':''D:/SAR-bagnet/saved_model/model.pth''}
model_dir='D:/SAR-bagnet/saved_model'
```   
  The above code is in the SAR_BagNet.py file,modify the corresponding code to correspond to your file location   
2.Replace the``` def forward(self,x)``` in SAR-Bagnet with the following code 
```
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits_list1 = []
        for i in range(N):
            for j in range(N):
                x1=x[:,:,i,j]
                x1.view(x1.size(0), -1)
                logits1=self.fc(x1)
                logits1=logits1[:,C]
                logits_list1.append(logits1.data.cpu().numpy().copy())
        logits2 = np.hstack(logits_list1)
        logits2 = logits2.reshape((N, N))
        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x,logits2   
```   
   N is the size of the heatmap, and C is the corresponding category of the heatmap 

3. run utils.py to generate heatmap 
##Author contact information  
If you have any questions, please contact me at 1441771519@qq.com
