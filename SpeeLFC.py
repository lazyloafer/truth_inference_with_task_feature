import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

class MaskedMultiCrossEntropy(object):

	def loss(self, y_true, y_pred):
		# print(y_true)
		# print(y_pred)
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, axis=1)
		mask = tf.equal(y_true[:,0,:], -1)
		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss

class CrowdsClassification(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", alpha=2, **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.alpha = alpha
        super(CrowdsClassification, self).__init__(**kwargs)

    def init_identities(self, shape, dtype=None):
        out = np.ones(shape)
        for r in range(shape[2]):
            for i in range(shape[0]):
                out[i, i, r] = self.alpha
        return out

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                          initializer=self.init_identities,
                                          trainable=True)
        elif self.conn_type == "VW":
            # vector of weights (one scale per class) per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                          initializer=keras.initializers.Ones(),
                                          trainable=True)
        elif self.conn_type == "VB":
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                               initializer=keras.initializers.Zeros(),
                                               trainable=True))
        elif self.conn_type == "VW+B":
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                               initializer=keras.initializers.Ones(),
                                               trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                               initializer=keras.initializers.Zeros(),
                                               trainable=True))
        elif self.conn_type == "SW":
            # single weight value per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.num_annotators, 1),
                                          initializer=keras.initializers.Ones(),
                                          trainable=True)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassification, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.conn_type == "MW":
            # task_embedding = x[:, :-1]
            # beta = self.lambda1 + self.lambda2 * tf.nn.sigmoid(x[:, -1])
            # print(x)
            # print(self.kernel)
            res = K.dot(x, self.kernel)
            # res = K.dot(x, tf.nn.softmax(self.kernel, axis=1))
        elif self.conn_type == "VW" or self.conn_type == "VB" or self.conn_type == "VW+B" or self.conn_type == "SW":
            out = []
            for r in range(self.num_annotators):
                if self.conn_type == "VW":
                    out.append(x * self.kernel[:, r])
                elif self.conn_type == "VB":
                    out.append(x + self.kernel[0][:, r])
                elif self.conn_type == "VW+B":
                    out.append(x * self.kernel[0][:, r] + self.kernel[1][:, r])
                elif self.conn_type == "SW":
                    out.append(x * self.kernel[r, 0])
            res = tf.stack(out)
            if len(res.shape) == 3:
                res = tf.transpose(res, [1, 2, 0])
            elif len(res.shape) == 4:
                res = tf.transpose(res, [1, 2, 3, 0])
            else:
                raise Exception("Wrong number of dimensions for output")
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.num_annotators)

class Run_LableMe():
    def __init__(self, BATCH_SIZE, N_EPOCHS, alpha):
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/LabelMe/prepared/"
        self.N_CLASSES = 8
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.data_train_vgg16, self.answers, self.answers_bin_missings, self.labels_train = self.load_LabelMe_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsClassification = CrowdsClassification(self.N_CLASSES, self.N_ANNOT, conn_type="MW", alpha=alpha)

    def load_LabelMe_dataset(self):

        def load_data(filename):
            with open(filename, 'rb') as f:
                data = np.load(f)
            return data

        print("\nLoading train data...")

        # images processed by VGG16
        data_train_vgg16 = load_data(self.DATA_PATH + "data_train_vgg16.npy")
        print(data_train_vgg16.shape)

        # ground truth labels
        labels_train = load_data(self.DATA_PATH + "labels_train.npy")
        print(labels_train.shape)

        # labels obtained from majority voting
        labels_train_mv = load_data(self.DATA_PATH + "labels_train_mv.npy")
        print(labels_train_mv.shape)

        # labels obtained by using the approach by Dawid and Skene
        labels_train_ds = load_data(self.DATA_PATH + "labels_train_DS.npy")
        print(labels_train_ds.shape)

        # data from Amazon Mechanical Turk
        print("\nLoading AMT data...")
        answers = load_data(self.DATA_PATH + "answers.npy")
        print(answers.shape)
        N_ANNOT = answers.shape[1]
        print("\nN_CLASSES:", self.N_CLASSES)
        print("N_ANNOT:", N_ANNOT)

        # load test data
        print("\nLoading test data...")

        # images processed by VGG16
        data_test_vgg16 = load_data(self.DATA_PATH + "data_test_vgg16.npy")
        print(data_test_vgg16.shape)

        # test labels
        labels_test = load_data(self.DATA_PATH + "labels_test.npy")
        print(labels_test.shape)

        print("\nConverting to one-hot encoding...")
        labels_train_bin = one_hot(labels_train, self.N_CLASSES)
        print(labels_train_bin.shape)
        labels_train_mv_bin = one_hot(labels_train_mv, self.N_CLASSES)
        print(labels_train_mv_bin.shape)
        labels_train_ds_bin = one_hot(labels_train_ds, self.N_CLASSES)
        print(labels_train_ds_bin.shape)
        labels_test_bin = one_hot(labels_test, self.N_CLASSES)
        print(labels_test_bin.shape)

        answers_bin_missings = []
        for i in range(len(answers)):
            row = []
            for r in range(N_ANNOT):
                if answers[i, r] == -1:
                    row.append(-1 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answers[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
        print(answers_bin_missings.shape)
        return data_train_vgg16, answers, answers_bin_missings, labels_train

    def build_base_model(self):
        base_model = Sequential()
        base_model.add(Flatten(input_shape=self.data_train_vgg16.shape[1:]))
        # base_model.add(Dense(1024, activation='relu'))
        base_model.add(Dense(128, activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Dense(self.N_CLASSES))
        base_model.add(Activation("softmax"))
        # if self.type == 'EM':
        #     base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                        loss='categorical_crossentropy')  ## for EM-NN
        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        model = self.build_base_model()

        model.add(self.CrowdsClassification)

        # instantiate specialized masked loss to handle missing answers
        loss = self.loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # compile model with masked loss and train
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(self.data_train_vgg16, self.answers_bin_missings, epochs=self.N_EPOCHS, shuffle=True, batch_size=self.BATCH_SIZE, verbose=1)

        # save weights from crowds layer for later
        # weights = model.layers[5].get_weights()

        # remove crowds layer before making predictions
        model.pop()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        accuracy_test = self.eval_model(model, self.data_train_vgg16, self.labels_train)
        print("Accuracy: Test: %.3f" % (accuracy_test,))

class Run_Music():
    def __init__(self, BATCH_SIZE, N_EPOCHS, alpha):
        self.DATA_PATH = "./dataset/music/"
        self.N_CLASSES = 10
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.task_feature, self.answers_bin_missings, self.truth = self.load_Music_dataset()
        self.N_ANNOT = self.answers_bin_missings.shape[-1]
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsClassification = CrowdsClassification(self.N_CLASSES, self.N_ANNOT, conn_type="MW", alpha=alpha)

    def load_Music_dataset(self):
        truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
        truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
        # print(truth)

        task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
        # print(task_feature_head)
        task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
        # print(task_feature)

        answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
        answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
        task_num = max(answers[:, 0]) + 1
        worker_num = max(answers[:, 1]) + 1
        answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

        for i in range(answers.shape[0]):
            answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
        # print(answer_matrix[-1])
        answers_bin_missings = []
        for i in range(len(answer_matrix)):
            row = []
            for r in range(worker_num):
                if answer_matrix[i, r] == -1:
                    row.append(-1 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
        # print(answers_bin_missings.shape)
        return task_feature, answers_bin_missings, truth

    def build_base_model(self):
        base_model = Sequential()
        # base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dense(256, activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN

        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        ##CL-MW
        model = self.build_base_model()

        model.add(self.CrowdsClassification)

        # instantiate specialized masked loss to handle missing answers
        loss = MaskedMultiCrossEntropy().loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # compile model with masked loss and train
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(self.task_feature, self.answers_bin_missings, epochs=self.N_EPOCHS, shuffle=True, batch_size=self.BATCH_SIZE, verbose=1)

        # save weights from crowds layer for later
        # weights = model.layers[5].get_weights()

        # remove crowds layer before making predictions
        model.pop()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        accuracy_test = self.eval_model(model, self.task_feature, self.truth)
        print("Accuracy: Test: %.3f" % (accuracy_test,))

class Run_SP():
    def __init__(self, BATCH_SIZE, N_EPOCHS, alpha):
        self.DATA_PATH = "./dataset/SP/"
        self.N_CLASSES = 2
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.task_feature, self.answers_bin_missings, self.truth = self.load_SP_dataset()
        self.N_ANNOT = self.answers_bin_missings.shape[-1]
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsClassification = CrowdsClassification(self.N_CLASSES, self.N_ANNOT, conn_type="MW", alpha=alpha)

    def load_SP_dataset(self):
        truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
        truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
        # print(truth)

        task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
        # print(task_feature_head)
        task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
        # print(task_feature)

        answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
        answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
        task_num = max(answers[:, 0]) + 1
        worker_num = max(answers[:, 1]) + 1
        answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

        for i in range(answers.shape[0]):
            answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
        # print(answer_matrix[-1])
        answers_bin_missings = []
        for i in range(len(answer_matrix)):
            row = []
            for r in range(worker_num):
                if answer_matrix[i, r] == -1:
                    row.append(-1 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
        # print(answers_bin_missings.shape)
        return task_feature, answers_bin_missings, truth

    def build_base_model(self):
        base_model = Sequential()
        # base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dense(256, activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN

        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        ##CL-MW
        model = self.build_base_model()

        model.add(self.CrowdsClassification)

        # instantiate specialized masked loss to handle missing answers
        loss = MaskedMultiCrossEntropy().loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # compile model with masked loss and train
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(self.task_feature, self.answers_bin_missings, epochs=self.N_EPOCHS, shuffle=True, batch_size=self.BATCH_SIZE, verbose=1)

        # save weights from crowds layer for later
        # weights = model.layers[5].get_weights()

        # remove crowds layer before making predictions
        model.pop()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        accuracy_test = self.eval_model(model, self.task_feature, self.truth)
        print("Accuracy: Test: %.3f" % (accuracy_test,))

class Run_BCD():
    def __init__(self, BATCH_SIZE, N_EPOCHS, alpha):
        self.DATA_PATH = "./dataset/BCD/"
        self.N_CLASSES = 2
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.task_feature, self.answers_bin_missings, self.truth = self.load_BCD_dataset()
        self.N_ANNOT = self.answers_bin_missings.shape[-1]
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsClassification = CrowdsClassification(self.N_CLASSES, self.N_ANNOT, conn_type="MW", alpha=alpha)

    def load_BCD_dataset(self):
        truth_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), nrows=0)
        truth = pd.read_csv('%s%s' % (self.DATA_PATH, 'truth.csv'), usecols=truth_head).values[:, -1]
        # print(truth)

        task_feature_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), nrows=0)
        # print(task_feature_head)
        task_feature = pd.read_csv('%s%s' % (self.DATA_PATH, 'task_feature.csv'), usecols=task_feature_head).values
        # print(task_feature)

        answers_head = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), nrows=0)
        answers = pd.read_csv('%s%s' % (self.DATA_PATH, 'answer.csv'), usecols=answers_head).values
        task_num = max(answers[:, 0]) + 1
        worker_num = max(answers[:, 1]) + 1
        answer_matrix = -1 * np.ones((task_num, worker_num), dtype=np.int32)

        for i in range(answers.shape[0]):
            answer_matrix[answers[i][0], answers[i][1]] = answers[i][2]
        # print(answer_matrix[-1])
        answers_bin_missings = []
        for i in range(len(answer_matrix)):
            row = []
            for r in range(worker_num):
                if answer_matrix[i, r] == -1:
                    row.append(-1 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
        # print(answers_bin_missings.shape)
        return task_feature, answers_bin_missings, truth

    def build_base_model(self):
        base_model = Sequential()
        # base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dense(256, activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(BatchNormalization(center=False, scale=False))
        base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN

        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        ##CL-MW
        model = self.build_base_model()

        model.add(self.CrowdsClassification)

        # instantiate specialized masked loss to handle missing answers
        loss = MaskedMultiCrossEntropy().loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # compile model with masked loss and train
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(self.task_feature, self.answers_bin_missings, epochs=self.N_EPOCHS, shuffle=True, batch_size=self.BATCH_SIZE, verbose=1)

        # save weights from crowds layer for later
        # weights = model.layers[5].get_weights()

        # remove crowds layer before making predictions
        model.pop()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        accuracy_test = self.eval_model(model, self.task_feature, self.truth)
        print("Accuracy: Test: %.3f" % (accuracy_test,))

# Run_LableMe(BATCH_SIZE=64, N_EPOCHS=50, alpha=2.0).run()
# Run_Music(BATCH_SIZE=700, N_EPOCHS=1000, alpha=4.7).run()
# Run_SP(BATCH_SIZE=5000, N_EPOCHS=1000, alpha=1.4).run()
Run_BCD(BATCH_SIZE=1000, N_EPOCHS=1000, alpha=1.4).run()

