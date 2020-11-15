# SPMF
Python implementation of the method proposed in
"Signed Network Representation by Preserving Multi-order Signed Proximity". Submitted to TKDE, under review.

## Overview
This repository is organised as follows:
- `input/` contains an example graph `WikiElec`;
- `output/` is the directory to store the learned node embeddings;
- `src/` contains the implementation of the proposed SLF method.

## Requirements
The implementation is tested under Python 3.7, with the folowing packages installed:
- `networkx==2.3`
- `numpy==1.16.5`
- `scikit-learn==0.21.3`
- `texttable==1.6.2`
- `tqdm==4.36.1`

## Input
The code takes an input graph in `.txt` format. Each row indicates an edge between two nodes separated by a `space` or `\t`. The file cannot contain a header. Nodes can be indexed starting with any non-negative number. The example graph - `WikiElec` - is donwloaded from [SNAP](http://snap.stanford.edu/data/#signnets, but node ID is resorted). The structure of the input file is the following:

| Source node | Target node | Sign |
| :-----:| :----: | :----: |
| 0 | 1 | -1 |
| 1 | 3 | 1 |
| 1 | 2 | 1 |
| 2 | 4 | -1 |

**NOTE** `SPMF` is tested on **directed** networks. However, it can also handle undirected networks.

## Options
#### Input and output options
```
--edgePath               STR      Input file path                           Default=="./input/WikiElec.txt"
--sourceRepPath          STR      Source representation path                Default=="./output/WikiElec_source"
--targetRepPath          STR      Target representation path                Default=="./output/WikiElec_target"
```
#### Model options
```
--dim                    INT      Dimension of the representation           Default==32
--k                      INT      Number of noise samples                   Default==5
--h                      INT      Highest order considered                  Default==5
--sliceSize              INT      Slice size for computing summary matrix   Default==1000
--filter                 BOOL     Use filter trick or not                   Default==False
--r                      INT      Parameter of filter trick                 Default==5
```
#### Evaluation options
```
--testsize               FLOAT    Test ratio                                Default==0.2
--splitSeed              INT      Random seed for splitting dataset         Default==1
--linkPrediction         BOOL     Perform link prediction or not            Default=False
--signPrediction         BOOL     Perform sign prediction or not            Default=True
```

## Examples
Perform `SPMF` on the deafult `WikiElec` dataset, output the performance on the sign prediction task, and save the embeddings:
```
python src/main.py
```

Perform `SPMF` with custom test ratio and split seed:
```
python src/main.py --splitSeed 5 --test-size 0.3
```

If you want to learn node embedding for other use and not to waste time performing the link prediction or sign prediction tasks, then run:
```
python src/main.py --link-prediction False --sign-prediction False
```

## Output

#### Tasks on signed networks
For **sign prediction** task, we use `AUC` and `Macro-F1` for evaluation.

For **link prediction** task, we use `AUC` with OvO scheme, `Macro-F1` for evaluation.

An example output is like the following:
```
Calculating summary matrix: 100%|█████████████████████████████████████| 8/8 [00:06<00:00,  1.17it/s]
Evaluating...
Sign prediction: AUC 0.883, F1 0.768
Link prediction: AUC 0.896, F1 0.641
```

#### Node representations
The learned representations are saved in `output/` in `.npy` format (supported by `Numpy`). Note that if the maximal node ID is 36, then the embedding matrix has 36+1 rows ordered by node ID (as the ID can start from 0). Although some nodes may not exist (e.g., node 11 is removed from the original dataset), it does not matter.

You can use them for any purpose in addition to the two performed tasks.
