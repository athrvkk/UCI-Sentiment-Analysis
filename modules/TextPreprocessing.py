from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

class TextPreprocessing:

            def __init__(self, name):
                if name.lower() == 'amazon':
                    self.path = "../dataset/amazon_cells_labelled.txt"

                elif name.lower() == 'yelp':
                    self.path = "../dataset/yelp_labelled.txt"

                elif name.lower() == 'imdb':
                    self.path = "../dataset/imdb_labelled.txt"

                self.stop_words = stopwords.words('english')
                unwanted_stopwords = {'no', 'nor', 'not', 'ain', 'aren', "aren't", 'couldn', 'what', 'which', 'who',
                                      'whom',
                                      'why', 'how', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                                      'hasn',
                                      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                                      "wasn't",
                                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't"}

                self.stop_words = [ele for ele in self.stop_words if ele not in unwanted_stopwords]
                self.wordnet_lemmatizer = WordNetLemmatizer()
                self.embeddings_path = '../dataset/GloVe_Word_Embeddings/glove.6B.100d.txt'


            # ------------------------------------ READ DATA FROM .txt FILES ------------------------------------

            def get_data(self):
                file = open(self.path, "r")
                data = file.readlines()
                corpus = []
                labels = []

                for d in data:
                    d = d.split("\t")
                    corpus.append(d[0])
                    labels.append(d[1].replace("\n", ""))
                file.close()

                return corpus, labels

            # ------------------------------------ PREPROCESS TEXT ------------------------------------

            def preprocess_text(self, user_text):
                # Remove puntuations and numbers
                user_text = re.sub('[^a-zA-Z]', ' ', user_text)

                # Remove single characters
                user_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', user_text)

                # remove multiple spaces
                user_text = re.sub(r'\s+', ' ', user_text)
                user_text = user_text.lower()

                # Convert Text sentence to Tokens
                user_text = word_tokenize(user_text)

                # Remove unncecessay stopwords
                fintered_text = []
                for t in user_text:
                    if t not in self.stop_words:
                        fintered_text.append(t)

                # Word lemmatization
                processed_text1 = []
                for t in fintered_text:
                    word1 = self.wordnet_lemmatizer.lemmatize(t, pos="n")
                    word2 = self.wordnet_lemmatizer.lemmatize(word1, pos="v")
                    word3 = self.wordnet_lemmatizer.lemmatize(word2, pos=("a"))
                    processed_text1.append(word3)

                result = ""
                for word in processed_text1:
                    result = result + word + " "
                result = result.rstrip()

                return result

            # ------------------------------------- GENERATE COUNT FEATURES AS VECTORS---------------------------------

            def count_vectorize(self, X_train, X_test):
                count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
                count_vect.fit(X_train)

                # transform the training and validation data using count vectorizer object
                xtrain_count = count_vect.transform(X_train)
                xvalid_count = count_vect.transform(X_test)

                return xtrain_count,xvalid_count


            # ---------------------- GENERATE WORD LEVEL TF-IDF FEATURES AS VECTORS---------------------------------

            def word_TF_IDF_vectorize(self, X_train, X_test):
                tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
                tfidf_vect.fit(X_train)

                xtrain_tfidf = tfidf_vect.transform(X_train)
                xvalid_tfidf = tfidf_vect.transform(X_test)

                return xtrain_tfidf, xvalid_tfidf

            # ---------------------- GENERATE n-gram LEVEL TF-IDF FEATURES AS VECTORS---------------------------------

            def n_gram_TF_IDF_vectorize(self, X_train, X_test):
                tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                       max_features=10000)
                tfidf_vect_ngram.fit(X_train)

                xtrain_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
                xvalid_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

                return xtrain_tfidf_ngram, xvalid_tfidf_ngram

            # --------------------- GENERATE CHARACTER LEVEL TF-IDF FEATURES AS VECTORS------------------------------

            def char_TF_IDF_vectorize(self, X_train, X_test):
                tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                         max_features=10000)
                tfidf_vect_ngram_chars.fit(X_train)

                xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
                xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_test)

                return xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

            # ------------------------------------- TOKENIZE AND PAD FOR TRAINING--------------------------------------

            def tokenizer_and_pad_training(self, X_train, X_test, max_words, oov_word, padding_type, truncating_type, pad_len):
                # Generate Tokens sequences
                tokenizer = Tokenizer(num_words=max_words, oov_token=oov_word)
                tokenizer.fit_on_texts(X_train)
                X_train = tokenizer.texts_to_sequences(X_train)
                X_test = tokenizer.texts_to_sequences(X_test)

                # Pad the sequences
                vocab_size = len(tokenizer.word_index) + 2

                X_train_padded = np.array(pad_sequences(X_train, padding=padding_type, truncating=truncating_type, maxlen=pad_len))
                X_test_padded = np.array(pad_sequences(X_test, padding=padding_type, truncating=truncating_type, maxlen=pad_len))

                return tokenizer, vocab_size, X_train_padded, X_test_padded


            # ------------------------------------- TOKENIZE AND PAD FOR TRAINING--------------------------------------

            def preprocess_and_tokenize_test_case(self, tokenizer, test_case, padding_type, truncating_type, pad_len):
                processed_test_case = [self.preprocess_text(test_case)]
                instance = tokenizer.texts_to_sequences(processed_test_case)
                flat_list = []
                for sublist in instance:
                    for item in sublist:
                        flat_list.append(item)
                flat_list = [flat_list]
                instance = pad_sequences(flat_list, padding=padding_type, truncating=truncating_type, maxlen=pad_len)
                return instance

            # ------------------------------------- GET EMBEDDING MATRIX --------------------------------------

            def get_embedding_metrix(self, vocab_size, tokenizer):
                embeddings_dictionary = dict()
                glove_file = open(self.embeddings_path, encoding="utf8")

                for line in glove_file:
                    records = line.split()
                    word = records[0]
                    vector_dimensions = np.asarray(records[1:], dtype='float32')
                    embeddings_dictionary[word] = vector_dimensions
                glove_file.close()

                embedding_matrix = np.zeros((vocab_size, 100))
                for word, index in tokenizer.word_index.items():
                    embedding_vector = embeddings_dictionary.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[index] = embedding_vector

                return embedding_matrix


