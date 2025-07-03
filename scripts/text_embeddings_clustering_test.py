from datasets import load_dataset
import pickle
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import rootutils
from omegaconf import OmegaConf
from torchvision.transforms import RandAugment
import hydra
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA


ROOT = rootutils.setup_root("/home/phisch/multimodal", indicator=".project-root", pythonpath=True)

from src.analysis.torch_cka import CKA
from src.model.utils import get_model_and_processor
from src.model.model_module import LitMML, MLMWrapper
from src.data.data_module import MyDataModule


@hydra.main(version_base=None, config_path="configs", config_name="text_clustering")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(f"cuda:{cfg.device[0]}")

    model_config = cfg.model
    use_pretrained_model = cfg.get("use_pretrained_model", False)
    if use_pretrained_model:
        model_pretrained, _ = get_model_and_processor(model_config, pretrained=True)
        model_pretrained = MLMWrapper(model_pretrained)
        
    def get_ckpt_path(model_id):
        if cfg.checkpoints[model_id].get("path", None) is not None:
            ckpt_path = cfg.checkpoints[model_id].path
        elif model_id == "n5trwpk9":
            ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        elif model_id == "zathvtrx":
            ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        else:
            ckpt_dir = os.path.join(cfg.paths.ckpts_dir, model_id)
            ckpt_files = os.listdir(ckpt_dir)
            if len(ckpt_files) > 1 and "last.ckpt" in ckpt_files:
                ckpt_files.remove("last.ckpt")
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        return ckpt_path
    
    for model_id in cfg.checkpoints:
        ckpt_path = get_ckpt_path(model_id)
        model, processor = get_model_and_processor(model_config)
        model = MLMWrapper(model)
        lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                                model=model, 
                                                processor=processor,
                                                map_location=device,
                                                )
        model = lit_model.model
        print(f"Loaded checkpoint {model_id}")
        model_name = cfg.checkpoints[model_id].model
        
        # Lade AG News
        dataset_name = "ag_news"
        split="test"
        dataset = load_dataset(dataset_name, split=split)
        texts = [x["text"] for x in dataset]
        labels = [x["label"] for x in dataset]
        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        encodings = processor(text=texts, padding=True, truncation=True, return_tensors="pt")

        batch_size = 32
        dataloader = DataLoader(list(zip(encodings["input_ids"], encodings["attention_mask"])), batch_size=batch_size)

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model.basemodel.text_model(input_ids=input_ids, attention_mask=attention_mask)
                # Nimm z.B. den CLS-Token als Repräsentation
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
                all_embeddings.append(cls_embeddings.cpu())

        # Endgültige Matrix: (num_samples, hidden_dim)
        text_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        preds = kmeans.fit_predict(text_embeddings)

        nmi = normalized_mutual_info_score(labels, preds)
        ari = adjusted_rand_score(labels, preds)
        print(f"NMI: {nmi:.3f}, ARI: {ari:.3f}")

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(text_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=preds, cmap="tab10", alpha=0.7)
        plt.title("K-Means Clustering of Text Embeddings (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plot_path = f"test_results/kmeans_text/clustering_plot_{model_id}.png"
        plt.savefig(plot_path)

        # Save results to JSON
        import json
        results = {
            "model_id": model_id,
            "model_name": model_name,
            "dataset": dataset_name,
            "num_samples": len(texts),
            "num_clusters": n_clusters,
            "nmi": nmi,
            "ari": ari
        }
        results_path = f"test_results/kmeans_text/clustering_results_{model_id}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
