import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader


def plot_pred(pred, actual):
    sns.jointplot(pred, actual, kind='reg', joint_kws={'line_kws': {'color': 'green'}, 'scatter_kws': {"s": 2}})


def plot_loss(losses):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(losses)


def split_sample(dataset, validation_split=0.2, shuffle_dataset=True, random_seed=42, batch_size=128):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    return train_loader, validation_loader
