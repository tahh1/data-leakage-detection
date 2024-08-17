import pandas as pd
import torch
import dgl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report
from gnn import GNN
from graph_dataset import Dataset, handle_duplicate_ids
import argparse



def train(train_loader, val_loader, device, model,epochs):
    loss_fcn =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels,_) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.long().to(device) 
            batched_graph.ndata.pop("_ID")
            batched_graph.edata.pop("_ID")
            feat = batched_graph.ndata.pop("features").to(torch.float)
            labels = labels.flatten()
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
              "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                  epoch, total_loss / (batch + 1), train_acc, valid_acc
              )
          )


def evaluate(dataloader, device, model, evaluation_report=False):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    for batched_graph, labels, _ in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device).to(torch.float32).flatten()
        features = batched_graph.ndata.pop("features")
        
        total_samples += len(labels)
        logits = model(batched_graph, features)
        predicted = torch.max(logits, 1)[1].to(torch.float32).to(device)
        correct_predictions += (predicted == labels).sum().item()

    hamming_loss = 1.0 * correct_predictions / total_samples

    if evaluation_report:
        class_report = classification_report(labels.cpu(), predicted.cpu(),zero_division=0)
        print(class_report)
        class_report_dict = classification_report(labels.cpu(), predicted.cpu(), output_dict=True,zero_division=0)
        #print(confusion_matrix(labels.cpu(), predicted.cpu()))
        return hamming_loss, class_report_dict
    
    return hamming_loss

def cross_validate(n_splits, dataset, epochs, n_convs, hidden_dimensions, handle_duplicates, dataset_dup=None):
    labels = dataset.label
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    hamming_losses = []
    class_reports = []

    if handle_duplicates:
        print('Loading duplicate data information...')
        mappings, duplicate_indices = handle_duplicate_ids()

    for fold_index, (train_indices, val_indices) in enumerate(skf.split(indices, labels)):
        print(f'{"-" * 47} fold {fold_index + 1} {"-" * 47} ')

        if handle_duplicates:
            train_indices = [mappings[i] for i in train_indices]
            val_indices = [mappings[i] for i in val_indices]
            additional_indices = []
            for idx in val_indices:
                if idx in duplicate_indices.keys():
                    additional_indices.extend(duplicate_indices[idx])
            val_indices.extend(additional_indices)
            dataset = dataset_dup

        train_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_indices),
            batch_size=len(train_indices),
            pin_memory=torch.cuda.is_available()
        )

        val_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(val_indices),
            batch_size=len(val_indices),
            pin_memory=torch.cuda.is_available()
        )

        device = "cuda"if torch.cuda.is_available() else"cpu"
        input_size = dataset.feature_size
        output_size = dataset.num_labels
        model = GNN(input_size, hidden_dimensions, output_size, n_convs).to(device)

        train(train_loader, val_loader, device, model, epochs)
        ham_loss, class_report = evaluate(val_loader, device, model, evaluation_report=True)
        hamming_losses.append(ham_loss)
        class_reports.append(class_report)

    final_results = {
        "0.0": {"precision": 0, "recall": 0, "f1-score": 0},
        "1.0": {"precision": 0, "recall": 0, "f1-score": 0}
    }

    assert n_splits == len(class_reports)

    for report in class_reports:
        for label in ["0.0", "1.0"]:
            for metric in ["precision", "recall", "f1-score"]:
                final_results[label][metric] += report[label][metric] / n_splits

    print("-" * 44 + " Overall results" + "-" * 44 + "\n")
    print(f"Overall accuracy: {sum(hamming_losses) / len(hamming_losses):.4f}")
    print("Detailed metrics:\n")

    for label in ["0.0", "1.0"]:
        print(f"Metrics for label {label}:")
        for metric in ["precision", "recall", "f1-score"]:
            value = final_results[label][metric]
            print(f"  {metric.capitalize():<10}: {value:.4f}")
        print("\n")





def experiment_2(classifier):
    data_folder = './data/Experiment2'
    dataset_path_no_dup = './data/GitHub 1 NoDup.csv'
    dataset_path_dup = './data/GitHub 1.csv'

    print("Building the dataset for experiment 2...")
    dataset_dup = Dataset(dataset_path=dataset_path_dup, data_folder=data_folder, leakage=classifier)
    dataset_nodup = Dataset(dataset_path=dataset_path_no_dup, data_folder=data_folder, leakage=classifier)
    
    print(f"\nTraining {classifier} classifier for experiment 2...")
    cross_validate(5, dataset_nodup, 5, 6, 104, True, dataset_dup)
    
    print("Experiment 2 finished.")


def experiment_1(classifier):
    data_folder = './data/Experiment1'
    dataset = './data/Additional Labeled Data.csv'

    print("Building the dataset for experiment 1...")
    dataset = Dataset(dataset_path=dataset, data_folder=data_folder, leakage=classifier)

    print(f"\nTraining {classifier} classifier for experiment 1...")
    cross_validate(5, dataset, 5, 6, 104, False)
    
    print("Experiment 1 finished.")

def train_on_custom_dataset(data_folder, csv_file, classifier):
        print("Building the dataset...")
        dataset = Dataset(dataset_path=csv_file, data_folder=data_folder, leakage=classifier)
        print(f"\nTraining {classifier} classifier...")
        cross_validate(5, dataset, 5, 6, 104, False)


def main():
    parser = argparse.ArgumentParser(description="Run experiments with different classifiers. Specify an experiment number and a classifier or a folder path, csv path, and the classifier")
    parser.add_argument('--experiment', choices=['1', '2'], help="Choose which experiment to run: 1 or 2")
    parser.add_argument('--classifier', choices=['preprocessing', 'overlap'], required=True, help="Choose which classifier to train")
    parser.add_argument('--data-folder', help="Path to the data folder")
    parser.add_argument('--csv-file', help="Path to the CSV file")

    args = parser.parse_args()

    if args.experiment:
        if args.experiment == '1':
            experiment_1(args.classifier)
        elif args.experiment == '2':
            experiment_2(args.classifier)
    elif args.data_folder and args.csv_file:
        train_on_custom_dataset(args.data_folder, args.csv_file, args.classifier)
    else:
        print("Please provide, along with the classifier, either an experiment number or both data folder and CSV file path.")


if __name__ == "__main__":
    main()

