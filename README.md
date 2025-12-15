# MA-Net
MA Net is a network used for retinal vessel segmentation.

## Project Structure

```
MA-Net/
├── Datasets/          # Datasets
├── experiments/       # Experiment logs and results
├── lib/               # Common libraries and utility functions
│   ├── losses/        # Loss function implementations
│   ├── common.py      # Common functions
│   ├── dataset.py     # Data loaders
│   └── metrics.py     # Evaluation metrics
├── models/            # Network model implementations
│   ├── MA-Net.py      # Main network architecture
│   └── ...            # Other modules
├── prepare_dataset/   # Dataset preprocessing scripts
├── tools/             # Visualization tools
├── config.py          # Configuration file
├── requirements.txt   # Python dependencies
├── test.py            # Testing script
└── train.py           # Training script
```


## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare datasets:
Place datasets in the `Datasets/` directory, or modify path configurations in `config.py`

3. Train the model:
```bash
python train.py
```

4. Test the model:
```bash
python test.py
```

## Supported Datasets

- CHASEDB1
- DRIVE
- STARE

## Configuration Options

Modify `config.py` to adjust:
- Training parameters (epochs, batch size, etc.)
- Model parameters
- Dataset paths
- Experiment save paths
