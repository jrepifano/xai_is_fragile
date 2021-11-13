import os
import sys
import torch
import models
import influence
import numpy as np
import pytorch_lightning as pl


def finetune(model_type, top_40, test_idx, true_loss, batch_size):
    loss_diffs = list()
    for counter, idx in enumerate(top_40):
        if model_type == 'fc':
            model = models.fc(batch_size=batch_size, train_idx=idx, test_idx=test_idx)
        elif model_type == 'conv':
            model = models.conv(batch_size=batch_size, train_idx=idx, test_idx=test_idx)
        elif model_type == 'lenet':
            model = models.lenet(batch_size=batch_size, train_idx=idx, test_idx=test_idx)
        elif model_type == 'vgg13':
            model = models.vgg13(batch_size=batch_size, train_idx=idx, test_idx=test_idx)
        model.load_state_dict(torch.load(model_type+'.pt'))
        for param in model.parameters():
            param.requires_grad = False
        model.lin_last.weight.requires_grad = True
        model.lin_last.bias.requires_grad = True
        no_epochs = 30
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor='loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min',
            check_on_train_epoch_end=True
        )
        trainer = pl.Trainer(gpus='0', max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=100,
                         callbacks=[early_stop_callback])
        # trainer.tune(model)
        trainer.fit(model)
        loss_diffs.append(model.get_indiv_loss(model.test_dataloader()) - true_loss)
    return loss_diffs


def get_influence(model_type, test_idx=None, batch_size=1024):
    if model_type == 'fc':
        model = models.fc(batch_size=batch_size, test_idx=test_idx)
    elif model_type == 'conv':
        model = models.conv(batch_size=batch_size, test_idx=test_idx)
    elif model_type == 'lenet':
        model = models.lenet(batch_size=batch_size, test_idx=test_idx)
    elif model_type == 'vgg13':
        model = models.vgg13(batch_size=batch_size, test_idx=test_idx)
    else:
        raise ValueError('Invalid Model Type')
    model.load_state_dict(torch.load(model_type+'.pt'))
    i_up_loss = list()
    for itr, (x_test, y_test) in enumerate(model.test_dataloader()):
        pass
    infl = influence.influence_wrapper(model, None, None, x_test, y_test, model.train_dataloader())
    i_up_loss.append(infl.i_up_loss(model.lin_last.weight, estimate=True))
    i_up_loss = np.hstack(i_up_loss)
    return i_up_loss


def train(model_type, batch_size=1000, train_idx=None, test_idx=None):
    if model_type == 'fc':
        model = models.fc(batch_size=batch_size, train_idx=train_idx)
    elif model_type == 'conv':
        model = models.conv(batch_size=batch_size, train_idx=train_idx)
    elif model_type == 'lenet':
        model = models.lenet(batch_size=batch_size, train_idx=train_idx)
    elif model_type == 'vgg13':
        model = models.vgg13(batch_size=batch_size, train_idx=train_idx)
    no_epochs = 100
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min',
        check_on_train_epoch_end=True
    )
    trainer = pl.Trainer(gpus='0', max_epochs=no_epochs, auto_scale_batch_size='power', check_val_every_n_epoch=5,
                         callbacks=[early_stop_callback])
    # trainer.tune(model)
    trainer.fit(model)
    torch.save(model.state_dict(), model_type+'.pt')
    return model.get_losses(set='test')


def main(model_type='fc', gpu=0, num_samples=10):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    est_loss_diffs = list()
    true_loss_diffs = list()
    for i in range(num_samples):
        batch_size = 1024
        test_losses = train(model_type, batch_size=batch_size)
        max_loss = np.argsort(test_losses)[-1]
        # max_loss = np.argsort(test_losses)[len(test_losses) // 2] # median test point
        true_loss = test_losses[max_loss]
        i_up_loss = get_influence(model_type, max_loss, batch_size=batch_size)
        top_40 = np.argsort(np.abs(i_up_loss))[::-1][:40]
        est_loss_diffs.append(i_up_loss[top_40])
        true_loss_diffs.append(finetune(model_type, top_40, max_loss, true_loss, batch_size=batch_size))
        np.save('est_loss_diffs.npy', est_loss_diffs, allow_pickle=True)
        np.save('true_loss_diffs.npy', true_loss_diffs, allow_pickle=True)
        print('{}/{}'.format(i, num_samples))


if __name__ == '__main__':
    try:
        model_type = str(sys.argv[1])
        gpu = int(sys.argv[2])
        num_samples = int(sys.argv[3])
        main(model_type, gpu, num_samples)
    except Exception:
        main()
