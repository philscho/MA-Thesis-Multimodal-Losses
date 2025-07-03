from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature

def clip_contrastive_loss(image_out, text_out, temperature):
    # credits : https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
    # TODO make temperature learnable
    logits = (text_out @ image_out.T) / temperature
    images_similarity = image_out @ image_out.T
    texts_similarity = text_out @ text_out.T
    targets = torch.nn.functional.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    texts_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    images_loss = torch.nn.functional.cross_entropy(
        logits.T, targets.T, reduction="none"
    )
    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)

    return loss

def contrastive_loss(image_features, text_features):
    image_batch = torch.nn.functional.normalize(image_features, dim=-1)
    text_batch = torch.nn.functional.normalize(text_features, dim=-1)
    logits = torch.dot(image_batch, text_batch.T)   
    pass

def image_text_matching_loss(image_embeds, text_embeds, i2t_sim, t2i_sim) -> torch.float:
    bs = image_embeds.size(0)
    neg_image_embeds, neg_text_embeds = _neg_embeddings(
        image_embeds, text_embeds, i2t_sim, t2i_sim)
    selection = torch.randint(0, 2, (bs,)).unsqueeze(1).to(image_embeds.device)
    selected_text_embeds = torch.where(selection == 0, text_embeds, neg_text_embeds)
    multimodal_embeds = torch.concat((image_embeds, selected_text_embeds), dim=1)

    logits = self.itm_head(multimodal_embeds)
    #probs = F.softmax(logits, dim=1)
    loss_matching = self.matching_loss(logits, selection.float())
    self.log(f"loss-train-matching", loss_matching, sync_dist=True)
    losses_all.append(loss_matching)

# Source: https://github.com/facebookresearch/multimodal/blob/e4d288b45b89cee462a21ab264405f3f368adc21/torchmultimodal/models/albef/model.py#L293
def _neg_embeddings(
        image_embeds: Tensor,
        text_embeds: Tensor,
        text_atts: Tensor,
        similarity_i2t: Tensor,
        similarity_t2i: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg, text_atts_neg

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch import nn


def cosine(x: Tensor) -> Tensor:
    """
    Computes the pairwise cosine similarity between all vectors in `x`.

    Formula:
        `cosine(x, y) = (x . y) / (||x||_2 * ||y||_2)`

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector or
        a 3D tensor of shape `[b, n, d]` where `b` is the batch size.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` containing the pairwise cosine distance between all vectors in `x` or
        a 3D tensor of shape `[b, n, n]` containing the pairwise cosine distance between all vectors in `x`.
    """
    norm = x / x.norm(p=2, dim=-1, keepdim=True)
    return norm.matmul(norm.transpose(-2, -1))


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss as proposed in SimCLR paper.

     L = -log exp(sim(z_i, z_j) / tau) / sum_{k=1}^{2N} exp(sim(z_i, z_j)  / tau)
       = -sim(z_i, z_j) / tau + log sum_{k=1}^{2N} exp(sim(z_i, z_j)  / tau)
    """

    def __init__(self,
                 temperature: float = 0.07,
                 learn_temperature: bool = False,
                 gather_distributed: bool = True):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, requires_grad=learn_temperature))
        self.sim = cosine
        self.gather_distributed = gather_distributed
        self.sim_matrix = None
        self.logits = None

    def forward(self, x: Tensor, x_pair: Tensor, pl_module: LightningModule, **kwargs) -> Tensor:
        """
        Calculate the loss given a set of features and their augmented pairs.

        Parameters
        ----------
        x : Tensor
            The first set of features. Shape: (B, D)
        x_pair : Tensor
            The second set of features. Shape: (B, D)
        pl_module : LightningModule
            The PyTorch Lightning module that is using this loss function to handle distributed training.

        Example Similarity Matrix
        -----------------
        * A,B are from device 1
        * C,D are from device 2
        * A2 is the augmented version of A1, ...
           | A1 B1 C1 D1 | A2 B2 C2 D2 |
         A1| 0  X  X  X    P  X  X  X  |
         B1| X  0  X  X    X  P  X  X  |
         C1| X  X  0  X    X  X  P  X  |
         D1| X  X  X  0    X  X  X  P  |
           | ------------------------- |
         A2| P  X  X  X    0  X  X  X  |
         B2| X  P  X  X    X  0  X  X  |
         C2| X  X  P  X    X  X  0  X  |
         D2| X  X  X  P    X  X  X  0  |
        * P: Positive Sample
        * X: Negative Sample
        * 0: Self Similarity (ignored)
        """
        num_devices = pl_module.trainer.num_devices
        batch_size = x.shape[0]
        N = 2 * batch_size * num_devices

        if self.gather_distributed and num_devices > 1:
            x = pl_module.all_gather(x, sync_grads=True)  # (num_devices, B, D)
            x = x.view(-1, x.shape[-1])  # (B * num_devices, D)

            x_pair = pl_module.all_gather(x_pair, sync_grads=True)  # (num_devices, B, D)
            x_pair = x_pair.view(-1, x_pair.shape[-1])  # (B * num_devices, D)

        # concatenate the two sets of features
        x = torch.cat([x, x_pair], dim=0)  # x: (N, D)

        # calculate pairwise similarity matrix
        sim_mat = self.sim(x) / self.temperature  # sim_mat: (2B, 2B)
        #self.sim_matrix = sim_mat

        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim_mat, batch_size * num_devices)  # (B, 1)
        sim_j_i = torch.diag(sim_mat, -batch_size * num_devices)  # (B, 1)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # (2B, 1)

        # get the entries corresponding to positive and negative pairs
        logits = sim_mat.flatten()[1:].view(N - 1, N + 1)[:, :-1].reshape(N, N - 1)
        negative_loss = torch.logsumexp(logits, dim=1, keepdim=True)

        self.logits = logits
        return (-positive_samples + negative_loss).mean()
