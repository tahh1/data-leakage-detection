# Data Leakage Analysis
A deep learning and static analysis tool to detect test data leakage in Python notebooks.

This is the replication package for the TSE journal submission : [Learning Graph Representation of Machine
Learning Pipelines to Detect Data Leakage]()
The tool uses code and materials from the paper 
[Data Leakage in Notebooks: Static Detection and Better Processes](https://www.cs.cmu.edu/~cyang3/papers/ase22.pdf).
You can visit their repo [here](https://github.com/malusamayo/GitHubAPI-Crawler).


## How to build
1. Install [souffle](https://souffle-lang.github.io/install), the datalog engine we use for our main analysis. Make sure that souffle could be directly invoked in command line.
2. Pull and build the customized version of pyright, the type inference engine used by the previous approach: ```git submodule update --init --recursive ``` (please refer to the submodule for building the project).
3. Install required Python packages in requirements.txt. We use Python 3.8 for our tool; different Python versions might result in different parsed AST and unexpected errors. Additionally, make sure to install the appropriate DGL version for your environment (more information [here](https://www.dgl.ai/pages/start.html)).

## How to use
### Running the analysis and generating graphs
1. Run analysis for a single Python file: ```python3 -m src.main /path/to/file```. A new folder of the form ```file-fact``` will be created. It will have another folder named ```_graphs``` containing the different graph representations of pipelines for the different model pairs.
2. Run analysis for all Python files in a directory: ```python3 -m src.run /path/to/dir```
3. More information could be found using the `-h` flag.
 
### Training the classifiers
1. Run the analysis on the dataset folder (named as experiment 1 or 2). For simplicity, we only keep the notebooks containing model pairs and omit those that don't contain any from the previous dataset (experiment 2).
 ```bash
python -m src.run path/to/dataset_folder
```

2. Run the experiment of interest from the paper and specify the classifier to train (preprocessing or overlap)

```bash
python ./src/train.py --experiment <experiment_number> --classifier <classifier_type> 
```

3. Train the classifier using a given folder containing the training data given a ground truth

```bash
python ./src/train.py --data-folder path/to/train --csv-file path/to/csv --classifier <classifier_type> 
```


## How to build 
1. Pull the customized version of pyright: ```git submodule update --init --recursive```.
2. Add all used Python libraries to `requirements.txt`, which will be installed in the container and used by pyright.