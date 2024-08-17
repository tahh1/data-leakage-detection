from dgl.data.utils import load_graphs
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
from dgl.data import DGLDataset

def create_dataset(dataset,data_folder,leakage):
  labels=[]
  missed=[]
  graphs=[]
  names=[]
  for index,row in tqdm(dataset.iterrows()):
        file = row["File"]
        graph_of_interest=f'{data_folder}/{file}-fact/_graphs/{row["pair"]}.bin'

        if(os.path.exists(graph_of_interest)):
          try:
            graph , _= load_graphs(graph_of_interest)
            graphs.append(graph[0])
            if(leakage=="preprocessing"):
                label = [row['preproc GT']]                     
                labels.append(label)
            elif(leakage=="overlap"):
                label = [row['overlap GT']]                     
                labels.append(label)
            names.append(f'{file}/{row["pair"]}')
          except:
              continue
        else:
          missed.append(graph_of_interest)
  if(len(labels)==0):
      raise Exception("The specified folder do not contain the training data specified in the csv file")
  labels=torch.from_numpy(np.stack(labels,axis=0)).to(torch.float32)
  print(f"{len(missed)} model pairs could not be found")
  return graphs,labels,missed,names


def extend_dict_with_swaps(original_dict):
    """
    Extend the dictionary by creating additional entries where each value in the original lists 
    becomes a key, and the new value list is the original key and the other values in the list.
    
    :param original_dict: The original dictionary to extend.
    :return: The extended dictionary.
    """
    extended_dict = {}

    for key, values in original_dict.items():
        extended_dict[key] = values
        for value in values:
            if value not in extended_dict:
                extended_dict[value] = [key] + [v for v in values if v != value]

    return extended_dict

def map_model_pair_to_index(csv_file, model_pair):
    """Extract the id of a given model pair in the with-duplicate dataframe"""

    file, pair = extract_notebook_model_pair(model_pair)
    for idx, row in csv_file.iterrows():
        if row['File'] == file and row['pair'] == pair:
          return idx

def handle_duplicate_ids():
    #Load the full dataset and the one without duplicate pairs
    data_with_duplicates = pd.read_csv('./data/GitHub 1.csv', delimiter=";")
    data_without_duplicates = pd.read_csv('./data/GitHub 1 NoDup.csv', delimiter=";")

    #Load the identified duplicate pairs in the second experiment
    with open('./data/duplicates.json', 'r') as json_file:
        duplicate_pairs = json.load(json_file)

    #Map the index of a pair from the wihout-duplicate dataframe to its index in the with-duplicate dataframe
    mapping ={}
    for index,row in data_without_duplicates.iterrows():
        for index2,row2 in data_with_duplicates.iterrows():
            if(row['File']==row2['File'] and row['pair']==row2['pair'] ):
                mapping[index]=index2

    #Map the pair name to its index in the with-duplicate dataframe
    duplicate_idx= {map_model_pair_to_index(data_with_duplicates, k):[map_model_pair_to_index(data_with_duplicates, n) for n in v] for k,v in duplicate_pairs.items()}
    duplicate_idx = extend_dict_with_swaps(duplicate_idx)

    return mapping,duplicate_idx


def extract_notebook_model_pair(model_pair):
      """Extract the file and pair from the model pair path"""

      parts = model_pair.split('/')
      file = parts[-3]
      file = file.replace('-fact','')   
      pair  = parts[-1].replace('.bin', '')

      return file, pair

class Dataset(DGLDataset):
    _url = ''
    _sha1_str = ''
    def __init__(self, dataset_path ,data_folder,leakage,raw_dir=None,force_reload=False, verbose=False):
        self.dataset_path=dataset_path
        self.data_folder=data_folder
        self.leakage=leakage
        super(Dataset, self).__init__(name='Data Leakage dataset',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)


    def process(self):
        # process data to a list of graphs and a list of labels
        self.graphs, self.label,self.target_names = self._load_graph()
        

    def _load_graph(self):
        dataset = pd.read_csv(self.dataset_path,delimiter=";" )
        graphs,labels,_,target_names=create_dataset(dataset,self.data_folder,self.leakage)
        return graphs, labels, target_names

    @property
    def get_labels(self):
        return self.label

    @property
    def num_labels(self):
        return 2

    @property
    def feature_size(self):
        return self.graphs[0].ndata['features'].size()[1]

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx],self.target_names[idx]

    def __len__(self):
        return len(self.graphs)
    
