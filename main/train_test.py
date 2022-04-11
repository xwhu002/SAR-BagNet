import torch.utils.model_zoo as model_zoo
import torch
import time
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from main import SAR_BagNet, save


def train(train_iter, test_iter, net, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i,(X, y) in enumerate(train_iter):

            X = X.to(device)

            y = y.to(device)

            y_hat = net(X)

            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(test_iter, net)
        save.save_model_w_condition(model=net, model_dir='saved_model/', model_name=str(epoch) + 'bagnet_acc', accu=test_acc,
                                    target_accu=0.96)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
if __name__ == '__main__':
    mean = (0.0293905372581, 0.0293905372581, 0.0293905372581)
    std = (0.0308426998737, 0.0308426998737, 0.0308426998737)
    img_size=100
    train_dir = 'images/train'
    test_dir = 'images/test'
    train_batch_size=128
    test_batch_size=128
    normalize = transforms.Normalize(mean=mean,
                                       std=std)
    train_dataset = datasets.ImageFolder(
                    train_dir,
                    transforms.Compose([
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    normalize,
                    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    mode_dir = 'saved_model/'
    num_epochs=300
    net = SAR_BagNet.BagNet18(pretrained=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(train_loader,test_loader,net,optimizer,device,num_epochs)
    torch.save(obj=net, f=os.path.join(mode_dir, ('SARBagnet' + '{0:.3f}.pth').format(19)))