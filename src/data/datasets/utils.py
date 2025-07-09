import json

def imagenet_classnames(
        file_path='my_datasets/imagenet-simple-labels.json'
):
    ''' Returns a list of class names for the ImageNet dataset. '''
    with open(file_path) as f:
        labels = json.load(f)
    return labels

def imagenet_a_classnames(
        file_path='my_datasets/imagenet-a_class-names.json'
):
    ''' Returns a list of class names for the ImageNet-A dataset. '''
    with open(file_path) as f:
        labels = json.load(f)
    return labels
