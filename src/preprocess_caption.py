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

    def string_to_vector(self, caption) :
        """
        Converts captions to vector space representations using fasttext embeddings
        NOTE You have to execute the method `load_dataset` first, to load the fasttext embeddings
        :param The string, or list of strings to be transformed
        :return A N by 300 matrix, where N is the number words in the caption 
        """
        if not self.fasttext_loaded :
            raise "Fasttext is not yet loaded. Use the method 'load_fasttext' to load the word embeddings first."

        tokens = self.__tokenize(caption)

        word_vector = torch.Tensor([self.word_embedding.get_word_vector(word) for word in tokens])

        return word_vector

        
