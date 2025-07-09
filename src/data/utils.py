import json
from torch.utils.data import Subset, Dataset
import torchvision.datasets

def get_dataset_classnames(dataset):
    if type(dataset) == Subset: # for subset splits of a dataset
        dataset = dataset.dataset
        if type(dataset) == Subset:
            dataset = dataset.dataset
    name = dataset.__class__.__name__
    
    if name == "Places365":
        classnames = dataset.classes
        classnames = [x.replace('_', ' ').replace('-', ' ') for x in classnames]
    elif name == 'Caltech101' or name =='Caltech256':
        classnames = dataset.categories
        if name == 'Caltech256':
            classnames = [x.split('.')[1].replace('-', ' ').replace(' 101', '') for x in classnames]
        classnames = [x.replace('_', ' ').replace(' easy', '') for x in classnames]
    elif name == 'ImageNetValDataset':
        classnames = dataset.classnames
    elif name == 'Flowers102':
        classnames = []
        raise NotImplementedError
    elif name in ["Food101", "CIFAR100", ]:
        classnames = [x.replace('_', ' ') for x in dataset.classes]
    else:
        classnames = dataset.classes
    return classnames

def get_dataset_module(module_name):
    if module_name == "ImageNet":
        from src.data.datasets import ImageNetDataset
        dataset_module = ImageNetDataset
    else:
        dataset_module = getattr(torchvision.datasets, module_name, None)


class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 1