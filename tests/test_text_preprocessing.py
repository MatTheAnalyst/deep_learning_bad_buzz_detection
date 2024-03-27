from src.text_preprocessing import Text_preprocessing
import pandas as pd
import numpy as np

def test_should_clean_comment():
    # sut = system under test
    sut = Text_preprocessing()
    initial_comment = "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
    expected_comment = "awww that bummer you shoulda got david carr of third day to do it"
    assert sut.clean_comment(initial_comment) == expected_comment

def test_should_sample_dataset():
    sut = Text_preprocessing()
    initial_values = pd.Series(np.random.randint(100, size=(100)))
    initial_labels = pd.Series(np.random.randint(2, size=(100)))
    expected_sampling_shape = 10
    values_sample, labels_sample = sut.sampling(sample_size=10, values=initial_values, labels=initial_labels)
    assert len(values_sample) == 10

def test_should_lemmatize_comment():
    sut = Text_preprocessing()
    initial_comment = "I couldn't bear to watch it. And I thought the UA loss was embarrassing"
    expected_comment = "I could n't bear watch . And I think UA loss embarrass"
    assert sut.lemmatized_comment(initial_comment)[0] == expected_comment