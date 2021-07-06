import os
import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from influence import influence_wrapper
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, train_idx, test_idx, data_dir=os.getcwd(), batch_size=1024):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.test_idx = test_idx

    def setup(self, stage=None):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        self.mnist_test.data = self.mnist_test.data[self.test_idx].unsqueeze(0)
        self.mnist_test.targets = self.mnist_test.targets[self.test_idx].unsqueeze(0)
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)
        if self.train_idx != None:
            self.mnist_train.data = np.delete(self.mnist_train.data, self.train_idx, axis=0)
            self.mnist_train.targets = np.delete(self.mnist_train.targets, self.train_idx, axis=0)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class lenet(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
        self.count += len(y)
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
        print(self.count)
        self.count = 0

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True), 'monitor': 'loss'}
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


def train():
    model = lenet(batch_size=1024)
    no_epochs = 1000
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )
    trainer = pl.Trainer(gpus=1, max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=1, callbacks=[early_stop_callback])
    # trainer.tune(model)
    trainer.fit(model)
    torch.save(model.state_dict(), 'lenet.pt')
    model.eval()
    train_losses = model.get_losses(set='train')
    test_losses = model.get_losses(set='test')
    return train_losses, test_losses


def get_influence(test_idx):
    mnist = MNISTDataModule(None, test_idx)
    mnist.setup()
    model = lenet(batch_size=1024)
    model.load_state_dict(torch.load('lenet.pt'))
    i_up_loss = list()
    for itr, (x_test, y_test) in enumerate(mnist.test_dataloader()):
        pass
    for itr, (x_train, y_train) in enumerate(mnist.train_dataloader()):
        infl = influence_wrapper(model, x_train, y_train, x_test, y_test)
        i_up_loss.append(infl.i_up_loss(model.lin_last.weight, estimate=True))
    i_up_loss = np.hstack(i_up_loss)
    return i_up_loss


def finetune(top_40, test_idx, true_loss):
    loss_diffs = list()
    for counter, idx in enumerate(top_40):
        model = lenet(batch_size=1024)
        model.load_state_dict(torch.load('lenet.pt'))
        mnist = MNISTDataModule(idx, test_idx)
        no_epochs = 100
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0005,
            patience=5,
            verbose=True,
            mode='min'
        )
        trainer = pl.Trainer(gpus='0', max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=1,
                             callbacks=[early_stop_callback])
        trainer.fit(model, mnist)
        loss_diffs.append(model.get_indiv_loss(mnist.test_dataloader()) - true_loss)
        print('Done {}/{}'.format(counter + 1, len(top_40)))
    return loss_diffs
