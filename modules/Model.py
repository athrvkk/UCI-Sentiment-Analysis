from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# ------------------------------------------------------ TRAIN ML MODELS --------------------------------------------
def train_ml_model(classifier, X_train, X_test, y_train, y_test):
    # fit the training dataset on the classifier
    classifier.fit(X_train, y_train)
    # predict the labels on validation dataset
    predictions = classifier.predict(X_test)
    return classifier, metrics.accuracy_score(predictions, y_test)




# ------------------------------------------------------ CALLBACK CLASS --------------------------------------------

class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') >= 0.99):
            print("\nTraining accuracy above 99%...so training stopped...\n")
            self.model.stop_training = True



# ------------------------------------------------------ TRAIN DNN --------------------------------------------

def create_model_DNN(input_dim, embedding_dim, embedding_matrix, pad_len, trainable, n1=64, n2=32, kr=None, br=None):
    Simple = Sequential()
    embedding_layer = Simple.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                  trainable=trainable))
    Simple.add(GlobalMaxPooling1D())
    Simple.add(Dense(n1, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    Simple.add(Dense(n2, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    Simple.add(Dense(1, activation='sigmoid'))
    Simple.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
    return Simple



# ------------------------------------------------------ TRAIN CNN --------------------------------------------

def create_model_CNN(input_dim, embedding_dim, embedding_matrix, pad_len, trainable, n1, k, d=0.0, kr=None, br=None):
    myCNN = Sequential()
    myCNN.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                        trainable=trainable))
    myCNN.add(Conv1D(n1, kernel_size=k, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    myCNN.add(GlobalMaxPooling1D())
    myCNN.add(Dropout(d))
    myCNN.add(Dense(1, activation='sigmoid'))
    myCNN.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
    return myCNN



# ------------------------------------------------------ TRAIN RNN --------------------------------------------

def create_model_LSTM(input_dim, embedding_dim, embedding_matrix, pad_len, trainable, n1, n2, d=0.0):
    myLSTM = Sequential()
    myLSTM.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                  trainable=trainable))
    myLSTM.add(Bidirectional(LSTM(n1, dropout=d, return_sequences=True)))
    myLSTM.add(GlobalMaxPooling1D())
    myLSTM.add(Dense(n2, activation='relu'))
    myLSTM.add(Dense(1, activation='sigmoid'))
    myLSTM.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
    return myLSTM



# ----------------------------------------------- PLOT ACCURACY LOSS GRAPH --------------------------------------------

def plot_curves(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()



# ------------------------------------------------------ SAVE MODELS --------------------------------------------

def save_model(model, name):
    model.save("../models/DL/"+name+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open("../models/DL/"+name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../models/DL/"+name+"_weights.h5")


# ------------------------------------------------------ PREDICT LABELS --------------------------------------------

def predict_label(result):
    if result[0][0] >= 0.5:
        label='positive'
    else:
        label="negative"
    return label, result[0][0]
