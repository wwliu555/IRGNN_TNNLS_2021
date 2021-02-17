# IRGNN_TNNLS_2021
# IRGPR

This repo is a pytorch implementation of the paper "Item Relationship Graph Neural Networks for E-commerce", built upon [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library.

### Dependecies
* Python3.7
* PyTorch
* PyG
* networkx
* pandas
* gensim

### Experiment Data
* [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)
* To process a dataset from raw files
  * Please get the following files from [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/) and put them at the ```raw/``` directory.
  ```
  meta_Video_Games.json.gz
  reviews_Video_Games.json.gz
  ```
  * Obtain node features from reviews by ```gensim.models.doc2vec```, and put the ```.d2v``` file at ```raw/```.
* We provided a sample processed heterogenous graph ```Amazon_Video_Games.pt``` from Amazon Video Games raw data, so that you can directly load the processed data and train the model.

### Experiment
* Before running, please modify the corresponding Amazon data category in ```amazon_data_loader.py```.
```
python run_irgnn.py --lr [lr] --node_emb [node embedding dim]
```

