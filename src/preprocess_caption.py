import numpy as np
import fastText as ft
from nltk.tokenize import RegexpTokenizer
import torch

class PreprocessCaption :
    """
    Useful class to process captions in string form, to vector space representations
    """
    def __init__(self, fasttext_file) :
        self.fasttext_file = fasttext_file
        self.tokenizer = RegexpTokenizer("\\w+")
        self.fasttext_loaded = False
    
    def load_fasttext(self) :
        """
        Reads fasttext into memory 
        """
        print("Loading fasttext model...")
        self.word_embedding = ft.load_model(self.fasttext_file)
        print("done!")

        self.fasttext_loaded = True

    def __tokenize(self, sentence) :
        """
        Tokenizes a string sentence based on word contents
        :param sentence The sentence to tokenize
        :return A list of string tokens
        """
        return tokenizer.tokenize(sentence)

    def string_to_vector(self, caption, max_no_words=50) :
        """
        Converts captions to vector space representations using fasttext embeddings
        NOTE You have to execute the method `load_dataset` first, to load the fasttext embeddings

        :param The string, or list of strings to be transformed
        :param max_no_words Maximum number of words in the caption, final tensor will be zero-padded to this number
        :return A N by 300 matrix, where N is max_no_words. Also returns the number of actual words. 
        """
        if not self.fasttext_loaded :
            raise "Fasttext is not yet loaded. Use the method 'load_fasttext' to load the word embeddings first."

        tokens = self.__tokenize(caption)
        no_tokens = len(tokens)

        word_vector = torch.Tensor([self.word_embedding.get_word_vector(word) for word in tokens])

        # Pad the word vector so it always have the same dimensions
        if no_tokens < max_no_words :
            pad = torch.zeros(max_no_words - no_tokens, word_vector.shape[1])
            word_vector = torch.cat((word_vector, pad))

        return word_vector, no_tokens

        
