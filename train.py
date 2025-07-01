import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
from torchvision import datasets,transforms
from tqdm import tqdm
import os
from model.cnn import simplecnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.ImageFolder(root=os.path.join(r"/home/chief/edua/edua_chief/edua_chief/dataset","train"),transoforms=train_transformer) # 训练集做图像变换
testset = datasets.ImageFolder(root=os.path.join(r"/home/chief/edua/edua_chief/edua_chief/dataset","test"),transoforms=train_transformer) # 测试集做图像变换


# 定义训练集的加载器
train_loader = dataloader(trainset,batch_size=32,num_workers=0,shuffle=True)

# 定义测试集的加载器
test_loader = dataloader(testset,batch_size=32,num_workers=0,shuffle=False)

def train(model,train_loader,criterion,optimizer,num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs,labels in tqdm(train_loader,desc=f"epoch:{epoch+1}/{num_epochs}",unit="batch"):
            imputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs,labels) # loss计算
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item()* inputs.size(0) # 用loss乘批次大小 得到该批次的loss
        epoch_loss = running_loss/len(train_loader.dataset)
        print(f"epoch[{epoch+1}/{num_epochs},Train_loss{epoch_loss:.4f}]")

        accuracy = evaluate(model,test_loader,criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model,save_path)
            print("model saved with best acc",best_acc)

def evaluate(model,test_loader,criterion):
    model.eval()
    test_loss = 0.0
    correct =0
    total =0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device) # 将数据都送到设备里面
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            test_loss = test_loss + loss.item()* inputs.size(0)
            _ ,predicted = torch.max(outputs,1) # 获取模型预测的最大值
            total = total + labels.size(0) # 计算总样本的数量
            correct = correct + (predicted == labels).sum().item() # 正确样本数累加
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total # 计算准确率
    print(f"Test Loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}%")
    return accuracy

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path)

if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_class = 4
    save_path = "model_pth\best.pth"
    model = simplecnn(num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    train(model,train_loader,criterion,optimizer,num_epochs)
    evaluate(model,test_loader,criterion)