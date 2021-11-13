import os
import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torch.nn.utils import prune
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class GenericModel(pl.LightningModule):
    def __init__(self, batch_size, train_idx=None, test_idx=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.count = 0
        self.true_loss = None
        self.loss_diffs = list()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
        # self.count += len(y)
        # self.loss_diffs.append((self.criterion(self(self.x_test), self.y_test) - self.true_loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        acc = self.test_acc(logits.softmax(dim=-1), y)

    def on_validation_end(self):
        print('Train Acc {:.2f}, Test Acc: {:.2f}'.format(self.train_acc.compute(), self.test_acc.compute()))
        self.train_acc.reset()
        self.test_acc.reset()


    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        if self.train_idx != None:
            mnist_train.data = np.delete(mnist_train.data, self.train_idx, axis=0)
            mnist_train.targets = np.delete(mnist_train.targets, self.train_idx, axis=0)
            self.train_idx = None
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=2, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=False)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        if self.test_idx != None:
            mnist_test.data = mnist_test.data[self.test_idx].unsqueeze(0)
            mnist_test.targets = mnist_test.targets[self.test_idx].unsqueeze(0)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True), 'monitor': 'loss', 'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        # This just stops the version number from printing out on the progress bar. Not necessary to run the model.
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_losses(self, set='train'):
        if set == 'train':
            set = True
        elif set == 'test':
            set = False
        else:
            raise Exception('Invalid set')
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist = MNIST(os.getcwd(), train=set, download=True, transform=transform)
        dl = DataLoader(mnist, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=False)
        if self.test_idx != None:
            self.dl.data = self.mnist_test.data[self.test_idx].unsqueeze(0)
            self.dl.targets = self.mnist_test.targets[self.test_idx].unsqueeze(0)
        losses = list()
        for itr, (x, y) in enumerate(dl):
            logits = self.forward(x)
            losses.append(criterion(logits, y).detach().cpu().numpy())
        return np.hstack(losses)

    def get_indiv_loss(self, dl):
        for idx, (x, y) in enumerate(dl):
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            return loss.item()


class fc(GenericModel):
    def __init__(self, batch_size, train_idx=None, test_idx=None):
        super().__init__(batch_size, train_idx=None, test_idx=None)
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.lin_1 = torch.nn.Linear(1024, 128)
        self.lin_last = torch.nn.Linear(128, 10)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.Tanh()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.lin_1(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.flatten(x)
        x = self.relu(self.lin_1(x))
        return x

    def prune_model(self, percentage, model_name):
        parameter = ((self.lin_1, 'weight'),
                     (self.lin_last, 'weight'))
        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)



class conv(GenericModel):
    def __init__(self, batch_size, train_idx=None, test_idx=None):
        super().__init__(batch_size, train_idx=None, test_idx=None)
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.conv_1 = torch.nn.Conv2d(1, 4, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.lin_last = torch.nn.Linear(900, 10)
        # self.lin_last = torch.nn.Linear(128, 10)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.Tanh()

    def forward(self, x):
        x = self.pool(self.relu(self.conv_1(x)))
        x = self.flatten(x)
        # x = self.relu(self.lin_1(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.pool(self.relu(self.conv_1(x)))
        x = self.flatten(x)
        # x = self.relu(self.lin_1(x))
        return x


class lenet(GenericModel):
    def __init__(self, batch_size, train_idx=None, test_idx=None):
        super().__init__(batch_size, train_idx=None, test_idx=None)
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.conv_1 = torch.nn.Conv2d(1, 6, (5, 5))
        self.conv_2 = torch.nn.Conv2d(6, 16, (5, 5))
        self.conv_3 = torch.nn.Conv2d(16, 120, (5, 5))
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.tanh = torch.nn.Tanh()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(120, 84)
        self.lin_last = torch.nn.Linear(84, 10)
        self.count = 0
        self.true_loss = None
        self.loss_diffs = list()

    def forward(self, x):
        x = self.pool(self.tanh(self.conv_1(x)))
        x = self.pool(self.tanh(self.conv_2(x)))
        x = self.tanh(self.conv_3(x))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.pool(self.tanh(self.conv_1(x)))
        x = self.pool(self.tanh(self.conv_2(x)))
        x = self.tanh(self.conv_3(x))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        return x


class vgg13(GenericModel):
    def __init__(self, batch_size, train_idx=None, test_idx=None):
        super().__init__(batch_size, train_idx=None, test_idx=None)
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.conv1_1 = torch.nn.Conv2d(1, 64, (3, 3), padding='same')
        self.conv1_2 = torch.nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, (3, 3), padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc1 = torch.nn.Linear(512, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.lin_last = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        x = self.pool(self.relu(self.conv2_2(self.relu(self.conv2_1(x)))))
        x = self.pool(self.relu(self.conv3_2(self.relu(self.conv3_1(x)))))
        x = self.pool(self.relu(self.conv4_2(self.relu(self.conv4_1(x)))))
        x = self.pool(self.relu(self.conv5_2(self.relu(self.conv5_1(x)))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.pool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        x = self.pool(self.relu(self.conv2_2(self.relu(self.conv2_1(x)))))
        x = self.pool(self.relu(self.conv3_2(self.relu(self.conv3_1(x)))))
        x = self.pool(self.relu(self.conv4_2(self.relu(self.conv4_1(x)))))
        x = self.pool(self.relu(self.conv5_2(self.relu(self.conv5_1(x)))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.relu(self.fc3(x)))
        return x
