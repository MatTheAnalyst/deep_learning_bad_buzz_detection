from pathlib import Path 
import ssl 

def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    import nltk
    def download_nltk_data():
        packages = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for package in packages:
            nltk.download(package, download_dir=str(Path('env/nltk_data').absolute()))
    
    download_nltk_data()

setup_nltk()