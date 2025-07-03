import os
from pathlib import Path
import pickle
from typing import List, Union
import torch
from tqdm import tqdm
import numpy as np
import rootutils
ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from src.analysis import cka, rsa
from src.analysis.cka import CudaCKA
from src.utils.utils import save_as_pickle

def get_class_representations(
    dataloader,
    forward_func,
    num_classes,
    device="cuda",
    layers=None,  # NEW: list of layers to extract, e.g. [0, 6, 11, "last"]
):
    """
    Stores feature representations for each class and each specified layer in a dictionary.
    Args:
        dataloader: dataloader for the dataset
        forward_func: function to get the features from the model (should return all hidden states)
        num_classes: number of classes in the dataset
        device: device to use for computation
        layers: list of layers to extract (indices or "last")
    Returns:
        rep_dict: dict[layer][class] = list of representations
    """
    if layers is None:
        layers = ["last"]

    # rep_dict[layer][class] = list of features
    rep_dict = {layer: {i: [] for i in range(num_classes)} for layer in layers}

    for i, (examples, labels) in enumerate(tqdm(dataloader, desc="Getting features...")):
        # forward_func should return all hidden states as a tuple
        hidden_states = forward_func(examples.to(device))  # tuple: (embeddings, layer1, ..., last)
        # hidden_states: tuple of tensors, each [batch, seq, dim]
        for idx, lbl in enumerate(labels.flatten()):
            assert lbl < num_classes
            for layer in layers:
                if layer == "last" or layer == -1:
                    rep = hidden_states[-1][idx, 0, :]
                elif isinstance(layer, int):
                    rep = hidden_states[layer + 1][idx, 0, :]  # +1 because 0 is embeddings
                else:
                    raise ValueError(f"Unknown layer: {layer}")
                rep_dict[layer][lbl.item()].append(rep.detach().cpu())
    return rep_dict


def get_mean_class_representations_per_layer(
    dataloader,
    forward_func,
    num_classes,
    device="cuda",
    layers=None,  # e.g. [0, 6, 11, "last"]
):
    """
    Computes mean class representations for each class and each specified layer, dynamically (online).
    Args:
        dataloader: dataloader for the dataset
        forward_func: function to get the features from the model (should return all hidden states)
        num_classes: number of classes in the dataset
        device: device to use for computation
        layers: list of layers to extract (indices or "last")
    Returns:
        mean_rep_dict: dict[layer][class] = mean representation (numpy array)
    """
    if layers is None:
        layers = ["last"]

    # Initialize running sums and counts
    sum_dict = {layer: {i: None for i in range(num_classes)} for layer in layers}
    count_dict = {layer: {i: 0 for i in range(num_classes)} for layer in layers}

    for i, (examples, labels) in enumerate(tqdm(dataloader, desc="Getting features...")):
        hidden_states = forward_func(examples.to(device))["hidden_states"]  # tuple: (embeddings, layer1, ..., last)
        for idx, lbl in enumerate(labels.flatten()):
            lbl = lbl.item()
            assert lbl < num_classes
            for layer in layers:
                if layer == "last" or layer == -1:
                    rep = hidden_states[-1][idx, 0, :]
                elif isinstance(layer, int):
                    rep = hidden_states[layer + 1][idx, 0, :]
                else:
                    raise ValueError(f"Unknown layer: {layer}")
                rep = rep.detach().cpu()
                if sum_dict[layer][lbl] is None:
                    sum_dict[layer][lbl] = rep.clone()
                else:
                    sum_dict[layer][lbl] += rep
                count_dict[layer][lbl] += 1

    # Compute mean for each class and layer
    mean_rep_dict = {}
    for layer in layers:
        reps = []
        for cls in range(num_classes):
            if count_dict[layer][cls] > 0:
                mean_rep = sum_dict[layer][cls] / count_dict[layer][cls]
                reps.append(mean_rep.numpy())
        if reps:
            mean_rep_dict[layer] = np.stack(reps, axis=0)  # shape: (num_classes, hidden_dim)
        else:
            mean_rep_dict[layer] = None

    return mean_rep_dict

def get_classname_representations(classnames, forward_func, device="cuda"):
    """ Stores feature representations for each class in a dictionary
    Args:
        dataloader: dataloader for the dataset
        forward_func: function to get the features from the model
        num_classes: number of classes in the dataset
        device: device to use for computation
    Returns:
        rep_dict: dictionary with class index as key and list of representations as value
    """
    forward_func(classnames)
    # rep_dict: list of class specific image features for every class index
    class_represenations = {i: [] for i in range(len(classnames))}
    for i, classname in enumerate(tqdm(classnames, desc="Getting features...")):
        feats = forward_func(classname).detach() # get image features
        class_represenations[i].append(feats) # 
    
    return class_represenations

def get_multimodal_class_representations(dataloader, 
                                        vision_forward_func, 
                                        text_forward_func,
                                        classnames, 
                                        device="cuda",
    ):
    """ Stores feature representations for each class in a dictionary
    Args:
        dataloader: dataloader for the dataset
        forward_func: function to get the features from the model
        num_classes: number of classes in the dataset
        device: device to use for computation
    Returns:
        rep_dict: dictionary with class index as key and list of representations as value
    """
    # rep_dict: list of class specific image features for every class index
    vision_class_represenations = {i: [] for i in range(len(classnames))}
    text_class_represenations = {i: [] for i in range(len(classnames))}
    for i, (examples, labels) in enumerate(tqdm(dataloader, desc="Getting features...")):
        vision_feats = vision_forward_func(examples.to(device)).detach() # get image features
        text_feats = text_forward_func(examples.to(device)).detach() # get text features
        for idx, lbl in enumerate(labels.flatten()):
            assert lbl < num_classes
            vision_class_represenations[lbl.item()].append(vision_feats[idx, ...]) # 
            text_class_represenations[lbl.item()].append(text_feats[idx, ...]) # 
    
    return vision_class_represenations, text_class_represenations


def get_average_class_representations(class_representations,
                                   device="cuda",
                                   save_file_path=None,
    ):
    """ Computes mean class representations for each class
    Args:
        class_representations: dictionary with class index as key and list of representations as value
        device: device to use for computation
        save_file_path: path to save the mean class representations
    Returns:
        mean_dict: dict of mean class representation for each class
    """
    mean_dict = [None] * len(class_representations)
    for cls in sorted(class_representations):
        mean_representation = torch.stack(class_representations[cls]).mean(0).flatten()
        # assert mean_vec.shape[0] == 512
        mean_dict[cls] = mean_representation.cpu().numpy()
    if save_file_path:
        save_as_pickle(mean_dict, save_file_path)
    
    return mean_dict


def get_rsa_matrix(reps, similarity_measure):
    """ Computes the RSA matrix for the given representations
    Args:
        reps: dictionary with class index as key and list of representations as value
        similarity_measure: similarity measure to use for RSA
    Returns:
        rsa_matrix: RSA matrix for the given representations
    """
    rsa_matrix = np.zeros((len(reps), len(reps)))
    for keyind1, key1 in enumerate(reps):
        for keyind2, key2 in enumerate(reps):
            rdm1 = rsa.get_rdm(reps[key1])
            rdm2 = rsa.get_rdm(reps[key2])

            rsaval = rsa.get_rsa(rdm1, rdm2, similarity_measure=similarity_measure)
            # print (rsaval)
            if similarity_measure != "riemann":
                rsa_matrix[keyind1, keyind2] = rsaval.statistic
            else:
                rsa_matrix[keyind1, keyind2] = rsaval
    
    return rsa_matrix

def _compute_cka_and_kernel_matrices(representations_per_model, save_file_path=None):
    reps = representations_per_model
    cka_mat = np.nan * np.zeros(
        (len(reps), len(reps))
    )
    kernel_mat = np.nan * np.zeros(
        (len(reps), len(reps))
    )
    for k1ind, key1 in enumerate(tqdm(reps, desc="Computing CKA and Kernel matrices...")):
        for k2ind, key2 in enumerate(reps):
            if np.isnan(cka_mat[k2ind, k1ind]) == True:
                ckaval = cka.linear_CKA(reps[key1], reps[key2])
                cka_mat[k1ind, k2ind] = ckaval
                cka_mat[k2ind, k1ind] = ckaval

                kernelckaval = cka.kernel_CKA(reps[key1], reps[key2])
                kernel_mat[k1ind, k2ind] = kernelckaval
                kernel_mat[k2ind, k1ind] = kernelckaval
    
    if save_file_path:
        save_as_pickle({"cka_mat": cka_mat, "kernel_mat": kernel_mat},
                    save_file_path
        )

    return cka_mat, kernel_mat


def _compute_cka_and_kernel_matrices_cuda(representations_per_model, representations_per_model_2=None, device="cuda", save_file_path=None):
    cka = CudaCKA(device)
    reps = representations_per_model
    if representations_per_model_2 is not None:
        reps_2 = representations_per_model_2
        assert len(reps) == len(reps_2)
    else:
        reps_2 = representations_per_model
    cka_mat = torch.nan * torch.zeros(
        (len(reps), len(reps))
    ).to(device)
    kernel_mat = torch.nan * torch.zeros(
        (len(reps), len(reps))
    ).to(device)
    for k1ind, key1 in enumerate(tqdm(reps, desc="Computing CKA and Kernel matrices...")):
        for k2ind, key2 in enumerate(reps_2):
            if torch.isnan(cka_mat[k2ind, k1ind]) == True:
                x = torch.from_numpy(reps[key1]).to(device).to(torch.float32)
                y = torch.from_numpy(reps_2[key2]).to(device).to(torch.float32)
                ckaval = cka.linear_CKA(x, y)
                cka_mat[k1ind, k2ind] = ckaval
                cka_mat[k2ind, k1ind] = ckaval

                kernelckaval = cka.kernel_CKA(x, y)
                kernel_mat[k1ind, k2ind] = kernelckaval
                kernel_mat[k2ind, k1ind] = kernelckaval
    
    cka_mat = cka_mat.cpu().numpy()
    kernel_mat = kernel_mat.cpu().numpy()

    if save_file_path:
        save_as_pickle({"cka_mat": cka_mat, "kernel_mat": kernel_mat},
                    save_file_path
        )
    return cka_mat, kernel_mat

def compute_cka_and_kernel_matrices(representations_per_model, save_file_path=None, device=None):
    if "cuda" in device:
        print("Using CUDA for CKA computation")
        return _compute_cka_and_kernel_matrices_cuda(representations_per_model, device, save_file_path)
    else:
        print("Using CPU for CKA computation")
        return _compute_cka_and_kernel_matrices(representations_per_model, save_file_path)

def compute_multimodal_cka_and_kernel_matrices(representations_img, representations_txt, save_file_path=None, device=None):
    if "cuda" in device:
        print("Using CUDA for CKA computation")
        return _compute_cka_and_kernel_matrices_cuda(representations_img, representations_txt, device, save_file_path)
    else:
        print("Using CPU for CKA computation")
        print("to be implemented")
        # return _compute_cka_and_kernel_matrices(representations_per_model, save_file_path)

def create_classname_embeddings(forward_func,
                                 classnames: List[str],
                                 templates: List = None,
                                 tokenizer=None,
                                 batch_size: int = 64,
                                 device: Union[str, torch.device] = "cuda",
                                 verbose: bool = False):
    templates = ['{}'] if templates is None else templates  # templates = ['a photo of a {}.']
    if isinstance(templates, str):
        templates = [templates]
    num_templates = len(templates)  # num_templates = 1
    batch_size = 2 ** ((batch_size // num_templates) - 1).bit_length() # batch_size = 2 ** ((64 // 1) - 1).bit_length() = 2 ** 6 = 64

    batch_class_names = None # batch_class_names = make_batches(classnames, batch_size) #TODO: fix this
    # if verbose:
    #     print ('batch_class_names : \n ',batch_class_names)
    with torch.no_grad():
        zeroshot_weights = []
        bar = tqdm(batch_class_names, desc="Classifier weights...") if verbose else batch_class_names
        for batch_class_name in bar:
            texts = [template.format(classname) for classname in batch_class_name for template in templates]
            # if verbose:
            #     print ('texts',texts)
            if tokenizer is not None:
                #texts = tokenizer(texts).to(device)  # tokenize Shape: batch_size * num_tokens x context_length
                input = tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt")
                texts = input['input_ids'].to(device)  # tokenize Shape: batch_size * num_tokens x context_length
                mask = input['attention_mask'].to(device)

            class_embeddings = forward_func(texts, mask)  # batch_size * num_tokens x embedding_dim
            #class_embeddings = forward_func(texts)  # batch_size * num_tokens x embedding_dim
            #class_embeddings = forward_func(input_ids=texts)  # batch_size * num_tokens x embedding_dim
            # forward_func(texts) => forward_func(input_ids=texts)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embeddings = class_embeddings.view(len(batch_class_name), num_templates,
                                                     -1)  # batch_size x num_tokens x embedding_dim
            class_embedding_mean = class_embeddings.mean(dim=1)  # batch_size x embedding_dim
            class_embedding_mean /= class_embedding_mean.norm(dim=1).view(-1, 1)

            zeroshot_weights.append(class_embedding_mean)
        zeroshot_weights = torch.concat(zeroshot_weights, dim=0).T
    return zeroshot_weights.to(device)


if __name__ == "__main__":
    device = torch.device("cuda:1")
    reps = {"model_1": np.random.randn(100, 64), "model_2": np.random.randn(100, 64)}
    cka_mat, kernel_mat = compute_cka_and_kernel_matrices(reps, device=device)
    print(cka_mat.shape)
    print(cka_mat)
    print(kernel_mat.shape)
    print(kernel_mat)
