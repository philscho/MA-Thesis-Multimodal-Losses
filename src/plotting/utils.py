import pandas as pd
import numpy as np

def load_scores_per_dataset_all_models_all_fractions(
        csv_path: str,
        method: str,
        method_notes: str,
        metric: str = "Top1Accuracy",
):
    scores = pd.read_csv(csv_path)
    selected_scores = scores.loc[(scores["method"] == method) &
                                (scores["method_notes"] == method_notes) &
                                (scores["metric"] == metric)]
    datasets = selected_scores["dataset"].unique()
    data_fractions = np.sort(selected_scores["dataset_fraction"].unique()) # sort fractions in ascending order
    print(data_fractions)
    data_fractions = data_fractions[data_fractions != "1-aug"]  # remove specific value (e.g., 0.0) from data_fractions
    models = selected_scores["model_name"].unique()
    
    plotting_scores = {} # {dataset: {model: [scores from 0.2 to full dataset]}}

    for dataset in datasets:
        dataset_scores = {}
        for model in models:
            model_scores = []
            for fraction in data_fractions:
                try:
                    score = scores.loc[(scores["dataset"] == dataset) & 
                                    (scores["model_name"] == model) & 
                                    (scores["dataset_fraction"] == fraction) &
                                    (scores["method"] == method) &
                                    (scores["metric"] == metric) &
                                    (scores["method_notes"] == method_notes), "score"].item()
                except:
                    print(f"Error: {dataset}, {model}, {fraction}")
                model_scores.append(score)
            dataset_scores[model] = model_scores
        plotting_scores[dataset] = dataset_scores

    return plotting_scores


def load_scores_per_dataset_one_fraction(
        csv_path: str,
        method: str,
        method_notes: str,
        metrics: list = ["Top1Accuracy", "Top3Accuracy", "Top5Accuracy"],
        dataset_fraction: float = 1.0,
):
    scores = pd.read_csv(csv_path)
    selected_scores = scores.loc[(scores["method"] == method) &
                                (scores["method_notes"] == method_notes) &
                                (scores["dataset_fraction"] == dataset_fraction) &
                                (scores["metric"].isin(metrics))]
    datasets = selected_scores["dataset"].unique()
    models = selected_scores["model_name"].unique()
    
    plotting_scores = {} # {dataset: {model: {metric: score}}}

    for dataset in datasets:
        dataset_scores = {}
        for model in models:
            model_scores = {}
            for metric in metrics:
                try:
                    score = scores.loc[(scores["dataset"] == dataset) & 
                                    (scores["model_name"] == model) & 
                                    (scores["dataset_fraction"] == dataset_fraction) &
                                    (scores["method"] == method) &
                                    (scores["metric"] == metric) &
                                    (scores["method_notes"] == method_notes), "score"].item()
                except:
                    print(f"Error: {dataset}, {model}, {dataset_fraction}")
                model_scores[metric] = score
            dataset_scores[model] = model_scores
        plotting_scores[dataset] = dataset_scores

    return plotting_scores


def zero_shot_text_layers_load_scores_per_dataset(
        csv_path: str,
        metrics: str = "Top1Accuracy",
        dataset_fraction: float = 1.0,
):
    scores = pd.read_csv(csv_path)
    selected_scores = scores
    # selected_scores = scores.loc[(scores["method"] == method) &
    #                             (scores["method_notes"] == method_notes) &
    #                             (scores["metric"].isin(metrics))]
    datasets = selected_scores["dataset"].unique()
    models = selected_scores["model_name"].unique()
    
    plotting_scores = {} # {dataset: {model: {metric: score}}}

    for dataset in datasets:
        dataset_scores = {}
        for model in models:
            model_scores = []
            for i in range(12):
                method = "zeroshot-text_layer_" + str(i)
                try:
                    score = scores.loc[(scores["dataset"] == dataset) & 
                                    (scores["model_name"] == model) &
                                    (scores["method"] == method) &
                                    (scores["metric"] == metrics), "score"].item()
                except:
                    print(f"Error: {dataset}, {model}, {method}")
                model_scores.append(score)
            dataset_scores[model] = model_scores
        plotting_scores[dataset] = dataset_scores

    return plotting_scores