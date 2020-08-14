from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import datasets
import numpy as np
import torch
import torchvision.transforms as tv_transforms

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000},
    "imagenet_32": {
        "train": 1281167,
        "test": 50000,
        "valid": 50050
    },
}
rng = np.random.RandomState(seed=1)

class SimpleDataset(Dataset):
    def __init__(self, dataset, transform=True):
        self.dataset=dataset
        self.transform=transform

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.dataset['labels'][index]
        if(self.transform):
            image = (image / 255. - 0.5) / 0.5
        return image, label, index

    def __len__(self):
        return len(self.dataset['images'])

class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

data_path = "./data"

def split_l_u(train_set, n_labels, n_unlabels, tot_class=6, ratio = 0.5):
    # NOTE: this function assume that train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // tot_class
    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // tot_class
    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
    for c in classes[tot_class:]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]
    return l_train_set, u_train_set

def split_test(test_set, tot_class=6):
    images = test_set["images"]
    labels = test_set['labels']
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)}

    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    return test_set

def load_mnist():
    splits = {}
    trans = tv_transforms.Compose([tv_transforms.ToPILImage(),tv_transforms.ToTensor(), tv_transforms.Normalize((0.5,), (1.0,))])
    for train in [True, False]:
        dataset = datasets.MNIST(data_path, train, transform=trans, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()

def load_cifar10():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR10(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits["train" if train else "test"] = data
    return splits.values()


def gcn(images, multiplier=55, eps=1e-10):
    #global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)

def get_dataloaders(dataset, n_labels, n_unlabels, n_valid, l_batch_size, ul_batch_size, test_batch_size, iterations,
                    tot_class, ratio):

    rng = np.random.RandomState(seed=1)

    if dataset == "MNIST":
        train_set, test_set = load_mnist()
        transform = False
    elif dataset == "CIFAR10":
        train_set, test_set = load_cifar10()
        train_set["images"] = gcn(train_set["images"])
        test_set["images"] = gcn(test_set["images"])
        mean, zca_decomp = get_zca_normalization_param(train_set["images"])
        train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
        test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
        # N x H x W x C -> N x C x H x W
        train_set["images"] = np.transpose(train_set["images"], (0, 3, 1, 2))
        test_set["images"] = np.transpose(test_set["images"], (0, 3, 1, 2))

        #move class "plane" and "car" to label 8 and 9
        train_set['labels'] -= 2
        test_set['labels'] -= 2
        train_set['labels'][np.where(train_set['labels'] == -2)] = 8
        train_set['labels'][np.where(train_set['labels'] == -1)] = 9
        test_set['labels'][np.where(test_set['labels'] == -2)] = 8
        test_set['labels'][np.where(test_set['labels'] == -1)] = 9

        transform = False

    #permute index of training set
    indices = rng.permutation(len(train_set['images']))
    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]

    #split training set into training and validation
    train_images = train_set['images'][n_valid:]
    train_labels = train_set['labels'][n_valid:]
    validation_images = train_set['images'][:n_valid]
    validation_labels = train_set['labels'][:n_valid]
    validation_set = {'images': validation_images, 'labels': validation_labels}
    train_set = {'images': train_images, 'labels': train_labels}


    #split training set into labeled and unlabeled data
    validation_set = split_test(validation_set, tot_class=tot_class)
    test_set = split_test(test_set, tot_class=tot_class)
    l_train_set, u_train_set = split_l_u(train_set, n_labels, n_unlabels, tot_class=tot_class, ratio=ratio)

    print("Unlabeled data in distribuiton : {}, Unlabeled data out distribution : {}".format(
          np.sum(u_train_set['labels'] < tot_class), np.sum(u_train_set['labels'] >= tot_class)))

    l_train_set = SimpleDataset(l_train_set, transform)
    u_train_set = SimpleDataset(u_train_set, transform)
    validation_set = SimpleDataset(validation_set, transform)
    test_set = SimpleDataset(test_set, transform)

    print("labeled data : {}, unlabeled data : {},  training data : {}".format(
        len(l_train_set), len(u_train_set), len(l_train_set) + len(u_train_set)))
    print("validation data : {}, test data : {}".format(len(validation_set), len(test_set)))
    data_loaders = {
        'labeled': DataLoader(
            l_train_set, l_batch_size, drop_last=True,
            sampler=RandomSampler(len(l_train_set), iterations * l_batch_size)),
        'unlabeled': DataLoader(
            u_train_set, ul_batch_size, drop_last=True,
            sampler=RandomSampler(len(u_train_set), iterations * ul_batch_size)),
        'valid': DataLoader(
            validation_set, test_batch_size, shuffle=False, drop_last=False),
        'test': DataLoader(
            test_set, test_batch_size, shuffle=False, drop_last=False)
    }
    return data_loaders
