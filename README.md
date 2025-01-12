# DNABART: A Genomic LLM Foundational Model for Sequence Correction and Classification

DNABART is a novel encoder-decoder transformer model adapted from BART architecture for genomic sequence analysis. It integrates bidirectional encoding capabilities with autoregressive decoding to enable both accurate sequence interpretation and generation.

## Key Features

- **Encoder-Decoder Architecture**: Leverages modified BART architecture with GELU activations for improved convergence
- **Efficient Tokenization**: Uses Byte-Pair Encoding (BPE) with a vocabulary size of 4096 for optimal genomic sequence representation
- **Robust Pretraining**: Implements denoising autoencoder approach with 30% nucleotide corruption rate
- **State-of-the-Art Performance**: Achieves SOTA results on 16 out of 26 GUE benchmark datasets
- **Resource Efficient**: Competitive performance with only 103M parameters (compared to 2.5B in NT and 117M in DNABERT2)

## Table of Contents
- [Installation](#installation)
- [Pretraining](#pretraining)
- [Finetuning](#finetune)
- [Acknowledgements](#acknowledgements)


## Installation

### Prerequisites
- Python 3.6+
- CUDA-compatible GPU (tested on V100 and H100)
- Git

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/dnabart.git
cd dnabart
```

2. Create and activate a virtual environment:
```bash
python -m venv dnabart_env
source dnabart_env/bin/activate  # On Windows: dnabart_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Weights & Biases Setup
1. Install wandb:
```bash
pip install wandb
```

2. Login to your W&B account:
```bash
wandb login
```

3. Configure W&B project (automatic during training)

### Data Preparation
1. Generate reference reads using wgsim:
```bash
# Make the script executable
chmod +x GenerateReads.sh
# Run the script
./GenerateReads.sh
```

2. Prepare tokenizer:
```bash
# For BPE tokenization
python tokenizer.py
```

### Directory Structure
Ensure the following directory structure exists:
```
dnabart/
├── checkpoints/
│   └── {corruption_type}/
├── reference_reads/
├── trained_models/
│   └── {corruption_type}/
└── genome_data/
    └── S288C_reference_sequence_R64-5-1_20240529.fsa
```


## Pretraining

### Setup

- Trained on Saccharomyces genome sequences (1M sequences)
- Multiple corruption strategies evaluated:
  - Substitution only
  - Deletion + Insertion
  - Substitution + Deletion + Insertion
- Optimized with AdamW (lr = 1e-4)
- Training: 3 epochs, batch size 5
- Hardware: 3x NVIDIA V100 32GB GPUs


### Training the Model

1. **Pretraining**
```bash
# Format:
python main.py <corruption_type> <encoding_type>

# Examples:
# For substitution corruption with BPE encoding
python main.py substitution bpe

# For insertion-deletion corruption with k-mer encoding
python main.py indel kmer
```

Command line arguments:
- `corruption_type`: Type of sequence corruption ['substitution', 'indel', 'both']
- `encoding_type`: Type of sequence encoding ['bpe', 'kmer']

2. **Inference**
```bash
python inference.py <corruption_type>
```

### Results

Best performance achieved with substitution-only corruption.

<img src="misc/pretraining_result.png" align="center" width="500"/>
