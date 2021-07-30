import os
import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from influence import influence_wrapper
from torch.utils.data import DataLoader, random_split


class vgg13(pl.LightningModule):
    def __init__(self, batch_size, train_idx_to_remove=None, test_idx=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_idx_to_remove = train_idx_to_remove
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
        # self.count += len(y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        acc = self.test_acc(logits.softmax(dim=-1), y)

    def on_validation_end(self):
        # print('Train Acc {:.2f}, Test Acc: {:.2f}'.format(self.train_acc.compute(), self.test_acc.compute()))
        self.train_acc.reset()
        self.test_acc.reset()

    # def on_train_epoch_end(self):
    #     print(self.count)
    #     self.count = 0

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        if self.train_idx_to_remove != None:
            mnist_train.data = np.delete(mnist_train.data, self.train_idx_to_remove, axis=0)
            mnist_train.targets = np.delete(mnist_train.targets, self.train_idx_to_remove, axis=0)
            self.train_idx_to_remove = None
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        if self.test_idx != None:
            mnist_test.data = mnist_test.data[self.test_idx].unsqueeze(0)
            mnist_test.targets = mnist_test.targets[self.test_idx].unsqueeze(0)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=False), 'monitor': 'loss', 'interval': 'epoch', 'frequency': 1}
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
        dl = DataLoader(mnist, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True)
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


def get_influence(test_idx, batch_size, gpu=0):
    model = vgg13(batch_size=batch_size, train_idx_to_remove=None, test_idx=test_idx)
    model.load_state_dict(torch.load('vgg.pt'))
    i_up_loss = list()
    for itr, (x_test, y_test) in enumerate(model.test_dataloader()):
        pass
    infl = influence_wrapper(model, None, None, x_test, y_test, model.train_dataloader(), gpu=gpu)
    i_up_loss.append(infl.i_up_loss(model.lin_last.weight, estimate=True))
    i_up_loss = np.hstack(i_up_loss)
    return i_up_loss


def finetune(gpu, top_40, test_idx, true_loss, batch_size):
    loss_diffs = list()
    for counter, idx in enumerate(top_40):
        model = vgg13(batch_size=batch_size, train_idx_to_remove=idx, test_idx=test_idx)
        model.load_state_dict(torch.load('vgg.pt'))
        for param in model.parameters():
            param.requires_grad = False
        model.lin_last.weight.requires_grad = True
        model.lin_last.bias.requires_grad = True

        no_epochs = 100
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor='loss',
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode='min',
            check_on_train_epoch_end=True
        )
        trainer = pl.Trainer(gpus=gpu, max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=100,
                             callbacks=[early_stop_callback])
        trainer.fit(model)
        loss_diffs.append(model.get_indiv_loss(model.test_dataloader()) - true_loss)
        # print('Done {}/{}'.format(counter + 1, len(top_40)))
    return loss_diffs


def train(gpu, batch_size):
    model = vgg13(batch_size=batch_size)
    no_epochs = 1000
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min',
        check_on_train_epoch_end=True
    )
    trainer = pl.Trainer(gpus=gpu, max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=1, callbacks=[early_stop_callback])
    # trainer.tune(model)
    trainer.fit(model)
    torch.save(model.state_dict(), 'vgg.pt')
