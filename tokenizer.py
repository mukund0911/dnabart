from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files='R1_true_sequences.txt', 
                vocab_size=50264, 
                min_frequency=2, 
                show_progress=True, 
                special_tokens=["<s>","<pad>","</s>","<unk>","<mask>",])

#Save the Tokenizer to disk
tokenizer.save_model('bpe_small_output')