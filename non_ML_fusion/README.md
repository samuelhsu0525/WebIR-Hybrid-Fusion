# WebIR Final Project

## Requirements
use BEIR from https://github.com/beir-cellar/beir.git

## Usage
### generate ranking list(use pre-trained model)
Run the main script and set fusion method and dataset: \
fusion method:
["weighted_sum", "rrf", "max_score", "min_score", "product_score"]

```bash
    python fusion.py --fusion_method method --dataset dataset_name
```