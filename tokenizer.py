import sys

from config import KMER_VOCAB
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


def bpe_tokenize():
    if not os.path.exists('bpe_tokenization'):
        os.mkdir('bpe_tokenization')

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train tokenizer on true sequences
    tokenizer.train(files=f'reference_reads/{corruption_type}/R1_true_sequences.txt', 
                    vocab_size=50264, 
                    min_frequency=2, 
                    show_progress=True, 
                    special_tokens=["<s>","<pad>","</s>","<unk>","<mask>",])

    #Save the Tokenizer
    tokenizer.save_model('bpe_tokenization')


def kmer_tokenize(sequence, max_length, k=3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    tokens = [KMER_VOCAB.get(kmer, KMER_VOCAB['<unk>']) for kmer in kmers]
    
    # Pad or truncate the sequence
    if len(tokens) < max_length:
        tokens = tokens + [KMER_VOCAB['<pad>']] * (max_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
        attention_mask = [1] * max_length
    
    return tokens, attention_mask

if __name__ == "__main__":
    enc_type = sys.argv[1]
    corruption_type = sys.argv[2]

    if enc_type == 'bpe':
        bpe_tokenize(corruption_type)
    elif enc_type == 'kmer':
        kmer_tokenize(corruption_type)
    else:
        print("Invalid encoding type. Choose 'bpe' or 'kmer'.")