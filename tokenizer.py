from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# import os
# os.mkdir('bpe_tokenization')

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer on true sequences
tokenizer.train(files=f'reference_reads_indel/R1_true_sequences.txt', 
                vocab_size=50264, 
                min_frequency=2, 
                show_progress=True, 
                special_tokens=["<s>","<pad>","</s>","<unk>","<mask>",])

#Save the Tokenizer
tokenizer.save_model('bpe_tokenization')