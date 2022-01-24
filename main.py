import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from TorchModel import torchmodel
#dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import cycle
from PIL import Image
import numpy as np
from torchvision import transforms

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
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)

model = torchmodel.viae_net(Xphy.shape[2], Xerr.shape[2], False)
model.init_weights()
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=0)
epochs = 1500

model.to(device=device)


decay_changer = 0

#training loop
for i in range(epochs):
    decay_changer += 1
    running_loss = 0.0
    for k, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        inputs[2] = inputs[2].to(device)

        labels = labels.to(device)
        optimizer.zero_grad(),
        outputs = model(inputs[0], inputs[1], inputs[2])

        

    if(decay_changer == 20):
        optimizer.param_groups[0]['weight_decay'] += 0.9