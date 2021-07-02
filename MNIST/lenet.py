import os
import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


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

    def forward(self, x):
        x = self.pool(self.tanh(self.conv_1(x)))
        x = self.pool(self.tanh(self.conv_2(x)))
        x = self.tanh(self.conv_3(x))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.pool(self.tanh(self.conv_1))
        x = self.pool(self.tanh(self.conv_2))
        x = self.pool(self.tanh(self.conv_3))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
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


def train():
    model = lenet(batch_size=1024)
    no_epochs = 100
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='loss',
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


def main():
    pass


if __name__ == '__main__':
    train()
