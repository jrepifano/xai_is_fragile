import os
import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class vgg13(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
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
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=3, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), transforms.Pad(2)])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': 'val_loss'}
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        # This just stops the version number from printing out on the progress bar. Not necessary to run the model.
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def main():
    model = vgg13(batch_size=1024)
    no_epochs = 20
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=2,
        verbose=False,
        mode='min'
    )
    trainer = pl.Trainer(gpus=1, max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=1, callbacks=[early_stop_callback])
    trainer.tune(model)
    trainer.fit(model)


if __name__ == '__main__':
    main()
