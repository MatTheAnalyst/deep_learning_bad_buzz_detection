import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re, string, yaml


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)


class Text_preprocessing():
    def __init__(self):
        pass

    def sampling(self, sample_size:int, values:pd.Series, labels:pd.Series, stratify:bool=True):
        if len(values) != len(labels):
            raise ValueError(f"Les tailles de 'values' ({len(values)}) et 'labels' ({len(labels)}) doivent être identiques.")
        
        if not isinstance(sample_size, int):
            raise TypeError("Sample_size doit être un entier.")
        
        min_label_count = labels.value_counts().min()
        if min_label_count < sample_size:
            raise ValueError(f"Sample_size ne peut pas dépasser {min_label_count}, le nombre minimal d'occurrences dans 'labels'.")

        if stratify:
            distinct_labels = labels.unique()
            labels_sample = pd.Series()
            for value in distinct_labels:
                labels_sample = pd.concat([labels_sample, labels[labels == value].sample(round(sample_size/len(distinct_labels)), random_state=42)])
    
        else:
            labels_sample = labels.sample(sample_size, random_state=42)
        
        values_sample = values.loc[labels_sample.index]

        return values_sample, labels_sample
    
    
    def clean_and_preprocess(self, data):
        """
        Nettoyer et prétraiter un commentaire ou une liste de commentaires.
        
        Args:
            data: Un commentaire (str) ou une liste de commentaires (liste de str).
        
        Returns:
            Le commentaire ou la liste de commentaires nettoyée(s).
        """

        def clean_str(comment):
            if not comment or not isinstance(comment, str):
                print(f"{comment} n'est pas un commentaire valide")
                return ""

            # Nettoyage du commentaire
            comment = re.sub(r'<.*?>', '', comment)
            comment = re.sub(r'[^\x00-\x7F]+', '', comment)
            comment = re.sub(r'[0-9]*', '', comment)
            comment = re.sub(r"(.)\\1{2,}", '', comment)
            url_username = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
            comment = re.sub(url_username, ' ', str(comment).lower()).strip()
            comment = comment.translate(str.maketrans('', '', string.punctuation))
            comment = ' '.join([word for word in comment.split() if len(word) > 1])

            return comment

        # Appliquer le nettoyage
        if isinstance(data, str):
            return clean_str(data.lower())
        elif isinstance(data, list):
            return [clean_str(comment.lower()) for comment in data]
        else:
            print("Type de donnée non pris en charge")
            return data

    def get_wordnet_pos(self, treebank_tag):
        """
        Convert treebank POS tags to WordNet POS tags.
        
        Args:
            treebank_tag: The POS tag from treebank.
        Returns:
            A WordNet POS tag corresponding to the given treebank tag.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    
    def delete_stopwords(self, tokens:list):
        stop_words = set(stopwords.words(config['nltk']['stop_word']))
        return [token for token in tokens if token not in stop_words]
    

    def lemmatized_tokens(self, tokens:list):
        """
        Lemmatize tokens based on their part-of-speech tags.
        
        Args:
            tokens: A list of word tokens.
        Returns:
            A list of lemmatized word tokens.
        """
        pos_tags = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(pos)) for word, pos in pos_tags]

    def tokenizing(self, data):
        """
        Tokenize the comment string into individual words.
        
        Args:
            comment: The comment string to be tokenized.
        Returns:
            A list of words obtained from the comment.
        """
        if isinstance(data, str):
            data_tokenized =  word_tokenize(data)
            data_tokenized_without_stopword = self.delete_stopwords(data_tokenized)
            data_lemmatized = self.lemmatized_tokens(data_tokenized_without_stopword)
        else:
            data_tokenized = data.apply(word_tokenize)
            data_tokenized_without_stopword = data.apply(self.delete_stopwords)
            data_lemmatized = data_tokenized_without_stopword.apply(self.lemmatized_tokens)
        
        return " ".join(data_lemmatized)

    def lemmatizing(self, data:pd.Series, delete_stopwords:bool = True):
        """
        Lemmatize tokens based on their part-of-speech tags.
        
        Args:
            tokens: A list of word tokens.
        Returns:
            A list of lemmatized word tokens.
        """
        if delete_stopwords:
            if isinstance(data, str):
                data_lemmatized = data.apply(self.delete_stopwords)
            else:
                data_lemmatized = data.apply(self.delete_stopwords)
        data_lemmatized = data
        
        return data_lemmatized.apply(self.lemmatized_tokens)
    
