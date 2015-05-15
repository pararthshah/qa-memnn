from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Embedding
from keras.layers.recurrent import LSTM

class LSTM:
    def __init__(self, n_words=None, n_embedding=64, lr=0.01, n_epochs=100, momentum=0.9, word_to_id=None):
        self.n_embedding = n_embedding
        self.lr = lr
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.n_words = n_words

        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

        self.model = Sequential()
        self.model.add(Embedding(self.n_words, self.n_embedding))
        self.model.add(LSTM(self.n_embedding, self.n_embedding, activation='tanh', inner_activation='hard_sigmoid'))
        self.model.add(Dense(self.n_embedding, 1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    def train(self, dataset, labels):
        self.model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)

    def predict(self, dataset, labels):
        score = self.model.evaluate(X_test, Y_test, batch_size=16)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    train_dataset, train_labels, word_to_id = parse_dataset(train_file)
    test_dataset, test_labels, _ = parse_dataset(test_file, word_id=len(word_to_id), word_to_id=word_to_id, update_word_ids=False)

    lstm = LSTM(n_words=len(word_to_id), word_to_id=word_to_id, n_epochs=10)
    lstm.train(train_dataset, train_labels)
    lstm.predict(train_dataset, train_labels)
    lstm.predict(test_dataset, test_labels)
