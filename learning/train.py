import math
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from torchsummary import summary
from tqdm import tqdm

from params import lr, epochs, batch_size, train_ratio, model_save_dir


def score_model(x, y, model, test_name="CNN"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # create the train dataset
    label_encoder = preprocessing.OneHotEncoder()
    y = label_encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=42)
    train_size, val_size = len(x_train), len(x_test)
    x_train = torch.Tensor(x_train)  # transform to torch tensor
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = TensorDataset(x_test, y_test)  # create your datset
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print("Model Summary: ")
    summary(model, input_size=x_train.shape[1:])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_loss = np.inf

    for epoch in range(epochs):
        mini_batch_i = 0
        batch_losses = []
        num_correct_preds = 0
        pbar = tqdm(total=math.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size),
                    desc='Training {}'.format(test_name))
        pbar.update(mini_batch_i)

        model.train()  # set the model in training model (dropout and batchnormal behaves differently in train vs. eval)
        for x, y in train_dataloader:
            optimizer.zero_grad()

            mini_batch_i += 1
            pbar.update(1)

            y_pred = model(x.to(device))
            y_pred = F.softmax(y_pred, dim=1)
            # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
            y_tensor = y.to(device)
            loss = criterion(y_tensor, y_pred)
            loss.backward()
            optimizer.step()

            # measure accuracy
            num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()

            pbar.set_description('Training [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))
        train_accs.append(num_correct_preds / train_size)
        pbar.close()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=math.ceil(len(val_dataloader.dataset) / val_dataloader.batch_size),
                        desc='Validating {}'.format(test_name))
            pbar.update(mini_batch_i := 0)
            batch_losses = []
            num_correct_preds = 0

            for x, y in val_dataloader:
                mini_batch_i += 1
                pbar.update(1)
                y_pred = model(x.to(device))
                y_pred = F.softmax(y_pred, dim=1)
                # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                y_tensor = y.to(device)
                loss = criterion(y_tensor, y_pred)
                pbar.set_description('Validating [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))
                batch_losses.append(loss.item())
                num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()

            val_losses.append(np.mean(batch_losses))
            val_accs.append(num_correct_preds / val_size)
            pbar.close()
        print(
            "Epoch {}: train accuracy = {:.8f}, train loss={:.8f}; val accuracy = {:.8f}, val loss={:.8f}".format(epoch,
                                                                                                                  train_accs[
                                                                                                                      -1],
                                                                                                                  train_losses[
                                                                                                                      -1],
                                                                                                                  val_accs[
                                                                                                                      -1],
                                                                                                                  val_losses[
                                                                                                                      -1]))

        if val_losses[-1] < best_loss:
            torch.save(model.state_dict(), os.path.join(model_save_dir, test_name))
            print(
                'Best model loss improved from {} to {}, saved best model to {}'.format(best_loss, val_losses[-1],
                                                                                        model_save_dir))
            best_loss = val_losses[-1]

        # Save training histories after every epoch
        training_histories = {'loss_train': train_losses, 'acc_train': train_accs, 'loss_val': val_losses,
                              'acc_val': val_accs}
        pickle.dump(training_histories, open(os.path.join(model_save_dir, 'training_histories.pickle'), 'wb'))

    plt.plot(training_histories['loss_train'])
    plt.plot(training_histories['loss_val'])
    plt.show()

