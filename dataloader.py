"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    num_in_train = 60000
    num_in_test = 10000

    def __init__(
            self,
            target_number=9,
            mean_bag_length=10,
            std_bag_length=2,
            num_bag=250,
            seed=1,
            train=True
    ):
        """
        target_numbers: int. positive digit value. Valid value is in [0, 9].
        mean_bag_length: int. mean value of size of bag.
        std_bag_length: float. std value of size of bag. This value is used as std of normal dist.
        num_bag: int. The number of bags
        seed: int. numpy's seed value.
        train: bool. flag value whether train or test.
        """

        assert target_number in range(0, 10)
        assert mean_bag_length > 1
        assert std_bag_length > 0.
        assert num_bag >= 1

        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.std_bag_length = std_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        if self.train:
            self.num_samples = self.num_in_train
        else:
            self.num_samples = self.num_in_test

        self.bags_list, self.labels_list = self._create_bags(train)

    def _create_bags(self, train: bool):
        loader = data_utils.DataLoader(
            datasets.MNIST(
                '../datasets',
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]
                )
            ),
            batch_size=self.num_samples,
            shuffle=False
        )

        # fetch all data samples at once
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.std_bag_length, 1))
            bag_length = max(bag_length, 1)

            indices = torch.LongTensor(self.r.randint(0, self.num_samples, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = (labels_in_bag == self.target_number).type(torch.FloatTensor)

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = [max(self.labels_list[index]), self.labels_list[index]]

        return bag, label


if __name__ == "__main__":
    target_numbers = 9
    mean_bag_length = 10
    std_bag_length = 2
    num_bags = 100
    seed = 1
    batch_size = 1

    train_loader = data_utils.DataLoader(MnistBags(target_number=target_numbers,
                                                   mean_bag_length=mean_bag_length,
                                                   std_bag_length=std_bag_length,
                                                   num_bag=num_bags,
                                                   seed=seed,
                                                   train=True),
                                         batch_size=batch_size,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=target_numbers,
                                                  mean_bag_length=mean_bag_length,
                                                  std_bag_length=std_bag_length,
                                                  num_bag=num_bags,
                                                  seed=seed,
                                                  train=False),
                                        batch_size=batch_size,
                                        shuffle=False)

    len_bag_list_train = []
    num_positive_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        num_positive_bags_train += label[0].numpy()[0]

    print(
        'Number positive train bags: {}/{}\n'
        'Number of instances per bag, mean: {}, min: {}, max {}\n'.format(
            num_positive_bags_train,
            len(train_loader),
            np.mean(len_bag_list_train),
            np.min(len_bag_list_train),
            np.max(len_bag_list_train))
    )

    len_bag_list_test = []
    num_positive_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        num_positive_bags_test += label[0].numpy()[0]

    print(
        'Number positive test bags: {}/{}\n'
        'Number of instances per bag, mean: {}, min: {}, max {}\n'.format(
            num_positive_bags_test,
            len(test_loader),
            np.mean(len_bag_list_test),
            np.min(len_bag_list_test),
            np.max(len_bag_list_test))
    )
