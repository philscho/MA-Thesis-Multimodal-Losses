from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

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
