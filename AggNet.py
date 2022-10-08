import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
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

class CrowdsCategoricalAggregator():

    def __init__(self, model, data_train, answers, batch_size=64, pi_prior=1.0):
        self.model = model
        self.data_train = data_train
        self.answers = answers
        self.batch_size = batch_size
        self.pi_prior = pi_prior
        self.n_train = answers.shape[0]
        self.num_classes = np.max(answers) + 1
        self.num_annotators = answers.shape[1]

        # initialize annotators as reliable (almost perfect)
        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))

        # initialize estimated ground truth with majority voting
        self.ground_truth_est = np.zeros((self.n_train, self.num_classes))
        for i in range(self.n_train):
            votes = np.zeros(self.num_annotators)
            for r in range(self.num_annotators):
                if answers[i, r] != -1:
                    votes[answers[i, r]] += 1
            self.ground_truth_est[i, np.argmax(votes)] = 1.0

    def e_step(self):
        print("E-step")
        for i in range(self.n_train):
            adjustment_factor = np.ones(self.num_classes)
            for r in range(self.num_annotators):
                if self.answers[i, r] != -1:
                    adjustment_factor *= self.pi[:, self.answers[i, r], r]
            # print(adjustment_factor)
            self.ground_truth_est[i, :] = np.transpose(adjustment_factor) * self.ground_truth_est[i, :]
        self.ground_truth_est = tf.one_hot(tf.argmax(self.ground_truth_est, axis=-1), depth=self.num_classes)

        return self.ground_truth_est

    def m_step(self, ):
        print("M-step")
        hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True, batch_size=self.batch_size, verbose=0)
        print(("loss:", hist.history["loss"][-1]))
        self.ground_truth_est = self.model.predict(self.data_train)
        # self.ground_truth_est = np.eye(self.num_classes)[np.argmax(self.ground_truth_est, axis=-1)]
        # print(self.ground_truth_est)

        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))
        for r in range(self.num_annotators):
            normalizer = np.zeros(self.num_classes)
            for i in range(self.n_train):
                if self.answers[i, r] != -1:
                    self.pi[:, self.answers[i, r], r] += np.transpose(self.ground_truth_est[i, :])
                    normalizer += self.ground_truth_est[i, :]
            normalizer = np.expand_dims(normalizer, axis=1)
            # print(normalizer)
            self.pi[:, :, r] = self.pi[:, :, r] / np.tile(normalizer, [1, self.num_classes])
        # print(self.pi[:,:,r])
        # print('.............')

        return self.model, self.pi

class Run_LableMe():
    def __init__(self, BATCH_SIZE, N_EPOCHS):
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/LabelMe/prepared/"
        self.N_CLASSES = 8
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.data_train_vgg16, self.answers, self.answers_bin_missings, self.labels_train = self.load_LabelMe_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.model = self.build_base_model()
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsCategoricalAggregator = CrowdsCategoricalAggregator(self.model, self.data_train_vgg16, self.answers, batch_size=self.BATCH_SIZE, pi_prior=1.0)

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
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss='categorical_crossentropy')  ## for EM-NN
        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        crowds_agg = self.CrowdsCategoricalAggregator
        for epoch in range(self.N_EPOCHS):
            print("Epoch:", epoch + 1)

            # E-step
            ground_truth_est = crowds_agg.e_step()
            print("Adjusted ground truth accuracy:",
                  1.0 * np.sum(np.argmax(ground_truth_est, axis=1) == self.labels_train) / len(self.labels_train))

            # M-step
            model, pi = crowds_agg.m_step()

            accuracy_test = self.eval_model(model, self.data_train_vgg16, self.labels_train)
            print("Accuracy: Test: %.3f" % (accuracy_test,))

class Run_Music():
    def __init__(self, BATCH_SIZE, N_EPOCHS):
        self.DATA_PATH = "./dataset/music/"
        self.N_CLASSES = 10
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.task_feature,  self.answer_matrix, self.answers_bin_missings, self.truth = self.load_Music_dataset()
        self.N_ANNOT = self.answers_bin_missings.shape[-1]
        self.model = self.build_base_model()
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsCategoricalAggregator = CrowdsCategoricalAggregator(self.model, self.task_feature, self.answer_matrix, batch_size=self.BATCH_SIZE, pi_prior=1.0)

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
        return task_feature, answer_matrix, answers_bin_missings, truth

    def build_base_model(self):
        base_model = Sequential()
        # base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
        base_model.add(Dense(64, activation='relu'))
        base_model.add(Dense(256, activation='relu'))
        # base_model.add(Dropout(0.5))
        base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                           loss='categorical_crossentropy')  ## for EM-NN

        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        crowds_agg = self.CrowdsCategoricalAggregator
        for epoch in range(self.N_EPOCHS):
            print("Epoch:", epoch + 1)

            # E-step
            ground_truth_est = crowds_agg.e_step()
            print("Adjusted ground truth accuracy:",
                  1.0 * np.sum(np.argmax(ground_truth_est, axis=1) == self.truth) / len(self.truth))

            # M-step
            model, pi = crowds_agg.m_step()

            accuracy_test = self.eval_model(model, self.task_feature, self.truth)
            print("Accuracy: Test: %.3f" % (accuracy_test,))

class Run_SP():
    def __init__(self, BATCH_SIZE, N_EPOCHS):
        self.DATA_PATH = "./dataset/SP/"
        self.N_CLASSES = 2
        self.N_EPOCHS = N_EPOCHS
        self.task_feature,  self.answer_matrix, self.answers_bin_missings, self.truth = self.load_Music_dataset()
        self.BATCH_SIZE = BATCH_SIZE
        self.N_ANNOT = self.answers_bin_missings.shape[-1]
        self.model = self.build_base_model()
        self.loss = MaskedMultiCrossEntropy().loss
        self.CrowdsCategoricalAggregator = CrowdsCategoricalAggregator(self.model, self.task_feature, self.answer_matrix, batch_size=self.BATCH_SIZE, pi_prior=1.0)

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
        return task_feature, answer_matrix, answers_bin_missings, truth

    def build_base_model(self):
        base_model = Sequential()
        # base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
        base_model.add(Dense(64, activation='relu'))
        base_model.add(Dense(256, activation='relu'))
        # base_model.add(Dropout(0.5))
        base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss='categorical_crossentropy')  ## for EM-NN

        return base_model

    def eval_model(self, model, test_data, test_labels):
        # testset accuracy
        preds_test = model.predict(test_data)
        preds_test_num = np.argmax(preds_test, axis=1)
        accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)

        return accuracy_test

    def run(self):
        crowds_agg = self.CrowdsCategoricalAggregator
        for epoch in range(self.N_EPOCHS):
            print("Epoch:", epoch + 1)

            # E-step
            ground_truth_est = crowds_agg.e_step()
            print("Adjusted ground truth accuracy:",
                  1.0 * np.sum(np.argmax(ground_truth_est, axis=1) == self.truth) / len(self.truth))

            # M-step
            model, pi = crowds_agg.m_step()

            accuracy_test = self.eval_model(model, self.task_feature, self.truth)
            print("Accuracy: Test: %.3f" % (accuracy_test,))

# Run_LableMe(BATCH_SIZE=64, N_EPOCHS=50).run()

# #*#
# Run_Music(BATCH_SIZE=32, N_EPOCHS=1000).run()
# #*#

Run_SP(BATCH_SIZE=64, N_EPOCHS=50).run()