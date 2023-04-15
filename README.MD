# Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

# Configuration
There is a file named `confg.json` in the root directory.
You can change the parameters in this file to train the model.

This file is a list of configs for each dataset.

for example:
```json
    [
        {
        "seed": 4,
        "dataset": "PEMS04",
        "dataset_path": "data/PEMS04/pems04.npz",
        "num_node": 307,
        "embed_dim": 10,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "horizon": 12,
        "hidden_dim": 64,
        "num_layers": 2,
        "k": 2,
        "batch_size": 64
    },
    ...
    ]
```

- `seed` is the random seed for training.
- `dataset` is the name of the dataset, this is used to organize the results.
- `dataset_path` is the path of the dataset.
- `num_node` is the number of nodes in the graph.
- `embed_dim` is the dimension of the node embedding.
- `val_ratio` is the ratio of the validation set.
- `test_ratio` is the ratio of the test set.
- `horizon` is the maximum prediction horizon.
- `hidden_dim` is the dimension of the hidden state of the RNN.
- `num_layers` is the number of layers of the AGCRNLayer.
- `k` is the polynomial order (Chebyshev) of the AGCRNLayer.
- `batch_size` is the batch size of the training.

# Training
You can train the model by running the following command:
```bash
python train.py
```

The trained models will be saved in the `pretrained-model` directory.

# Evaluation
You can evaluate the model by running the following command:
```bash
python eval.py
```

This file reads the trained models in the `pretrained-model` directory and evaluates them.
Then writes back the metrics results to the `pretrained-model` directory as a json file.

# Visualization & Results
You can visualize the results by running the following command:
```bash
python plots.py
```
This file reads the metrics results in the `pretrained-model` directory and visualizes them.
There is a constant in the file called `DATASET` which is the name of the dataset you want to visualize.
It saves the figures in the `plots` directory.

Also it prints the data that we reported in the table for average on all datasets and all metrics.
For each dataset you have to change the `DATASET` constant in the file to the name of the dataset you want to output the results for.

# Running on GPU    
In both codes, train.py and eval.py, there is a line of code that looks like this:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = 'cuda:0'
```
You can change the number in the first line to the number of the GPU you want to use.
Also, you can change it to CPU by changing the second line to:
```python
device = 'cpu'
```

# Logs and Results
The logs will be printed in the console and in the `logs` directory as well, both for training and evaluation.
