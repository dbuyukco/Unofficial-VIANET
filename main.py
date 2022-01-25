import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from TorchModel import torchmodel
from torch.utils.tensorboard import SummaryWriter
#dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import cycle
from PIL import Image
import numpy as np
from torchvision import transforms
import time
import math

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(train_loader, model, criterion, optimizer, epoch):

    epoch_loss = 0.
    # switch to train mode
    model.train()
    iterator = 0
    end = time.time()
    for k, data in enumerate(train_loader, 0):
        inputs, label = data
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        inputs[2] = inputs[2].to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # compute output
        output = model(inputs[0], inputs[1], inputs[2])

        output_dim = output.shape[-1]
        output = output[0:].view(-1, output_dim)
        label = label[0:].view(-1)
        label = torch.reshape(label, [-1,1])

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iterator = k

    return epoch_loss / iterator


def evaluate(eval_loader, model, criterion):

    model.eval()

    iterator = 0
    epoch_loss = 0.
    with torch.no_grad():
        for k, data in enumerate(eval_loader, 0):
            inputs, label = data
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            inputs[2] = inputs[2].to(device)
            label = label.to(device)

            output = model(inputs[0], inputs[1], inputs[2])

            output_dim = output.shape[-1]
            output = output[0:].view(-1, output_dim)
            label = label[0:].view(-1)
            label = torch.reshape(label, [-1, 1])

            loss = criterion(output, label)
            iterator = k
            epoch_loss += loss.item()
    return epoch_loss/iterator

def create_sequence(dataset, length):
    data_sequences = []
    for index in range(len(dataset) - length):
        data_sequences.append(dataset[index: index + length])
    return np.asarray(data_sequences)

class timeseries(Dataset):
    def __init__(self, x, y, seq_len, imagePath, transform):
        self.pyhsX = torch.tensor(x_train[:,:,1:6], dtype=torch.float32)
        self.errX = torch.tensor(x_train[:,:,6:9], dtype=torch.float32)
        self.imageX = torch.tensor(x_train[:,:,0], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
        self.seq_len = seq_len
        self.imagePath = imagePath
        self.transform = transform
        self.convert_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        ##Split Dataset into Pyhsical, Error and Image function
        start = 0
        end = 0 + self.seq_len
        images = []
        for i in range(start,end):
            imgindx = int(self.imageX[idx, i])
            img_pth = self.imagePath + '\\' + str(imgindx) + '.png'
            img = Image.open(img_pth)
            if self.transform:
                img = self.transform(img)

            images.append(self.convert_tensor(img))
        imgX = torch.stack(images)
        return [imgX, self.pyhsX[idx], self.errX[idx]], self.y[idx]

    def __len__(self):
        return self.len

dataPath = r"C:\Users\TAI\Desktop\TrainSet"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataframe = pd.read_csv(dataPath + "\\" + "datawithAngle.csv", sep=",")
dataframe = dataframe[(dataframe['lambda'] != 0) & (dataframe['beta'] != 0)]

frameNames = dataframe['frame']
dataframe = dataframe.drop('frame', 1)

x = dataframe.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataframe = pd.DataFrame(x_scaled, columns=dataframe.columns)

Xphy = dataframe[['pitchrate','yawrate','rollrate','lambda','beta']]
Ximg = pd.DataFrame(frameNames)
Xerr = dataframe[['pitchrate','yawrate','rollrate']]
Y = dataframe[['laseraltitude']]

seq_len = 8
Xphy = create_sequence(Xphy,seq_len)
Ximg = create_sequence(Ximg,seq_len)
Xerr = create_sequence(Xerr,seq_len)

X = np.concatenate([Ximg, Xphy], -1)
X = np.concatenate([X, Xerr], -1)
Y = create_sequence(Y,seq_len)

# Split the remaining data to train and validation
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, shuffle=True)

print("X_train shape={}, and y_train shape={}".format(x_train.shape, y_train.shape))
print("X_test shape={}, and y_test shape={}".format(x_val.shape, y_val.shape))

img_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToPILImage()])
dataset = timeseries(x_train,y_train,seq_len=8, imagePath=dataPath, transform=img_transforms)
val_dataset = timeseries(x_val,y_val, seq_len=8, imagePath=dataPath, transform=img_transforms)
eval_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, pin_memory=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

model = torchmodel.viae_net(Xphy.shape[2], Xerr.shape[2], False)
model.init_weights()
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
epochs = 1500

model.to(device=device)

decay_changer = 0
best_valid_loss = float('inf')
#training loop
for i in range(epochs):
    print('EPOCH {}:'.format(i + 1))
    decay_changer += 1
    running_loss = 0.0

    if(decay_changer == 20):
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] += 0.9
        decay_changer = 0

    start_time = time.time()

    train_loss = train(train_loader=dataloader, model=model, criterion=criterion, optimizer=optimizer, epoch=i)
    val_loss = evaluate(eval_loader=eval_loader, model=model, criterion=criterion)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {i+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')





