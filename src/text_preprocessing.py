import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re, string


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
    
    def clean_comment(self, comment):
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
    
    def lemmatized_comment(self, comment):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Check format of comment
        if isinstance(comment, pd.Series):
            pass
        elif isinstance(comment, (str, list, dict)):
            comment = pd.Series(comment)
        else:
            print(f"Impossible d'appliquer une lemmatization.\nLe format du comment n'est pas reconnu : {type(comment)}")
        
        comment_tokenize = comment.apply(word_tokenize)
        comment_without_stopword = comment_tokenize.apply(lambda x: [x for x in x if x not in stop_words])
        comment_with_pos = comment_without_stopword.apply(pos_tag)
        comment_lemmatize = comment_with_pos.apply(lambda comment: [lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(pos)) for word, pos in comment])

        return comment_lemmatize.apply(lambda word: " ".join(word))

    
