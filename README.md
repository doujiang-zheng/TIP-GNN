- [TIP-GNN](#tip-gnn)
    - [Setup the environment](#setup-the-environment)
    - [Data Preprocess](#data-preprocess)
    - [Run the Script](#run-the-script)
        - [Temporal Link Prediction (Transductive)](#temporal-link-prediction-transductive)
        - [Temporal Link Prediction (Inductive)](#temporal-link-prediction-inductive)
        - [Temporal Node Classification](#temporal-node-classification)
    - [Cite us](#cite-us)

# TIP-GNN

> Authors: Tongya Zheng, Zunlei Feng, Tianli Zhang, Yunzhi Hao, Mingli Song, Xingen Wang, Xinyu Wang, Ji Zhao, Chun Chen

Code for [Transition Propagation Graph Neural Networks for Temporal Networks](https://ieeexplore.ieee.org/abstract/document/9955364).

## Setup the Environment

- `conda create -n tip python=3.9 -y`

- `pip install -r requirements.txt`

- My torch version is `torch-1.10.2+cu113`

## Data Preprocess

We have preprocessed most temporal graphs in the `data/format_data` directory, and placed the JODIE datasets at [Google drive](https://drive.google.com/drive/folders/19ItQ4G64rYa6so1IQ6NxEq_Ok7K9Sqsp?usp=sharing), which can be downloaded and placed at the `data/format_data`.

```bash
bash init.sh
```
We use `init.sh` to make necessary directories for our experiments to store generated datasets by `data/*`, boost the training speed by `gumbel_cache` and `sample_cache`, record training details by `log`, record testing results by `results` and `nc-results`, save our trained models by `ckpt` and `saved_models`.

```bash
python data_unify.py -t datasplit
python data_unify.py -t datalabel
```
We use `-t datasplit` to split datasets into the training, validation and testing set according to the ratios.

## Run the Script

### Temporal Link Prediction (Transductive)

In the setting of transductive temporal link prediction, we use trainable node embeddings.

- `python exper_edge_np.py -d fb-forum`

### Temporal Link Prediction (Inductive)

In the setting of inductive temporal link prediction,, we firstly generate features for each node to perform inductive link prediction.

- `python inductive_util.py -d fb-forum` 

- `python inductive_edge_np.py -d fb-forum`

### Temporal Node Classification

In the setting of temporal node classification prediction, we use the edge features and freeze the node embeddings as all zeros. 

Firstly, we have to train a pre-trained link prediction model following TGAT.

- `python exper_edge_np.py -d JODIE-wikipedia -t node -f`

- `python exper_node_np.py -d JODIE-wikipedia -f --balance --binary`

## Cite us
```
@article{zheng2022transition,
  title={Transition Propagation Graph Neural Networks for Temporal Networks},
  author={Zheng, Tongya and Feng, Zunlei and Zhang, Tianli and Hao, Yunzhi and Song, Mingli and Wang, Xingen and Wang, Xinyu and Zhao, Ji and Chen, Chun},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  pages={1--13},
  publisher={IEEE}
}
```
