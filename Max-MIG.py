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
import random

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def shuffle_data(train_data, answers_bin_missings, batch_size):
    data_num = train_data.shape[0]
    data_index = list(range(data_num))
    # random.shuffle(data_index)
    # if data_num % batch_size == 0:
    #     flag = int(data_num/batch_size)
    # else:
    #     flag = int(data_num / batch_size) + 1
    shuffle_train_data = train_data[data_index]
    shuffle_answers_bin_missings = answers_bin_missings[data_index]
    # for i in range(flag):
    return shuffle_train_data, shuffle_answers_bin_missings

class LableMe_model(tf.keras.Model):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/LabelMe/prepared/"
        self.N_CLASSES = 8
        self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_LabelMe_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.Dropout = Dropout(0.5)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.p_pure = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], dtype=np.float32)
        # self.p = self.p_pure

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
        # self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))


    def mig_loss_fuction(self, left_out, right_out):

        batch_num = left_out.shape[0]

        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1 )

        new_output = left_out / self.p_pure
        m = tf.matmul(new_output, right_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)
        m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
        # print(m)

        #loss 来自 KL，与MIG相反数
        return -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2

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
        self.data_test_vgg16 = load_data(self.DATA_PATH + "data_test_vgg16.npy")
        print(self.data_test_vgg16.shape)

        # test labels
        self.labels_test = load_data(self.DATA_PATH + "labels_test.npy")
        print(self.labels_test.shape)

        print("\nConverting to one-hot encoding...")
        labels_train_bin = one_hot(labels_train, self.N_CLASSES)
        print(labels_train_bin.shape)
        labels_test_bin = one_hot(self.labels_test, self.N_CLASSES)
        print(labels_test_bin.shape)

        answers_bin_missings = []
        for i in range(len(answers)):
            row = []
            for r in range(N_ANNOT):
                if answers[i, r] == -1:
                    row.append(0 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answers[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32) # task, worker, class
        # answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
        print(answers_bin_missings.shape)
        return data_train_vgg16, answers, answers_bin_missings, labels_train

    def identity_init(self, shape):
        out = np.ones(shape, dtype=np.float32) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        # sum_majority_prob = np.zeros((self.N_ANNOT, self.N_CLASSES))
        # expert_tmatrix = np.zeros(shape)
        #
        # for i in range(self.train_data.shape[0]):
        #     for j in range(self.answers_bin_missings.shape[0]):
        #         linear_sum_2 = np.sum(self.answers_bin_missings[j], axis=0)
        #         prob_2 = linear_sum_2 / np.sum(linear_sum_2)
        #         # prob_2 : all experts' majority voting
        #         for R in range(self.N_ANNOT):
        #             # If missing ....
        #             if max(self.answers_bin_missings[j, R]) == 0:
        #                 continue
        #             expert_class = np.max(np.max(self.answers_bin_missings[j, R]), 0)
        #             expert_tmatrix[R, :, int(expert_class)] += prob_2
        #             sum_majority_prob[R] += prob_2
        # sum_majority_prob = sum_majority_prob + 1 * (sum_majority_prob == 0)
        # for R in range(self.N_ANNOT):
        #     expert_tmatrix[R] = expert_tmatrix[R] / sum_majority_prob[R]

        return out

    def left_NN(self, input, training=None):
        flatten_input = self.flatten(input)
        flatten_input = self.bn(flatten_input)
        x = self.Dropout(self.fc1(flatten_input), training)
        x = self.bn1(x)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return cls_out

    def right_EM(self, crowd_answer, left_out, type=1, training=None):
        crowd_answer = tf.transpose(crowd_answer, (1,0,2))
        # print(crowd_answer)
        crowd_emb = tf.matmul(crowd_answer, self.kernel)
        agg_emb = tf.reduce_sum(crowd_emb, axis=0)
        out = 0
        if type == 1:
            # print(agg_emb.shape)
            # print(left_out.shape)
            # print(self.p_pure.shape)
            out = agg_emb + tf.math.log(left_out+0.001) + tf.math.log(self.p_pure)
        elif type == 2:
            out = agg_emb + tf.math.log(self.p_pure)
        elif type == 3:
            out = agg_emb + tf.math.log(left_out+0.001)
        return tf.nn.softmax(out, axis=-1)

    def call(self, task_feature, crowd_answer, training=None):
        left_out = self.left_NN(task_feature)
        right_out = self.right_EM(crowd_answer, left_out, type=1)
        return left_out, right_out

class Music_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/music/"
        self.N_CLASSES = 10
        self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_Music_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.Dropout = Dropout(0.5)

        self.p_pure = 0.125 * np.ones((self.N_CLASSES), dtype=np.float32)
        # self.p = self.p_pure

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))

    def mig_loss_fuction(self, left_out, right_out):

        batch_num = left_out.shape[0]

        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = left_out / self.p_pure
        m = tf.matmul(new_output, right_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)
        m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
        # print(m)

        #loss 来自 KL，与MIG相反数
        return -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2

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

        # Q = np.zeros((task_num, self.N_CLASSES))
        # for i in range(task_num):
        #     crowd_labels = answer_matrix[i]
        #     unique_crowd_labels = np.unique(crowd_labels)
        #     for j in range(len(unique_crowd_labels)):
        #         if unique_crowd_labels[j] == -1:
        #             continue
        #         else:
        #             count = len(np.where(crowd_labels == unique_crowd_labels[j])[0])
        #             Q[i, unique_crowd_labels[j]] = count
        # Q /= worker_num
        #
        # Pi = []
        # for i in range(worker_num):
        #     pi = np.zeros((self.N_CLASSES, self.N_CLASSES))
        #     worker_answers = answer_matrix[:, i]
        #     for j in range(self.N_CLASSES):
        #         Q_j = Q[:, j]
        #         for k in range(self.N_CLASSES):
        #             mask = np.where(worker_answers == k, 1, 0)
        #             if np.sum(Q_j) == 0:
        #                 # pi[j][k] = np.log(np.sum(Q_j * mask) / np.sum(Q_j))
        #                 pi[j][k] = 0
        #             else:
        #                 pi[j][k] = np.sum(Q_j * mask) / np.sum(Q_j)
        #     Pi.append(pi)
        # # Pi = np.concatenate(Pi, axis=-1)
        # Pi = np.array(Pi, dtype=np.float32)

        answers_bin_missings = []
        for i in range(len(answer_matrix)):
            row = []
            for r in range(worker_num):
                if answer_matrix[i, r] == -1:
                    row.append(0 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
        # print(answers_bin_missings.shape)
        return task_feature, answer_matrix, answers_bin_missings, truth

    def identity_init(self, shape):
        out = np.ones(shape, dtype=np.float32)
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 4.7
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 4.7
        return out
        # return self.Pi

    def left_NN(self, input, training=None):
        input = self.bn(input)
        x = self.Dropout(self.fc1(input), training)
        x = self.bn1(x)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return cls_out

    def right_EM(self, crowd_answer, left_out, type=1, training=None):
        crowd_answer = tf.transpose(crowd_answer, (1,0,2))
        # print(crowd_answer)
        crowd_emb = tf.matmul(crowd_answer, self.kernel)
        agg_emb = tf.reduce_sum(crowd_emb, axis=0)
        out = 0
        if type == 1:
            # print(agg_emb.shape)
            # print(left_out.shape)
            # print(self.p_pure.shape)
            out = agg_emb + tf.math.log(left_out+0.001) + tf.math.log(self.p_pure)
        elif type == 2:
            out = agg_emb + tf.math.log(self.p_pure)
        elif type == 3:
            out = agg_emb + tf.math.log(left_out+0.001)
        return tf.nn.softmax(out, axis=-1)

    def call(self, task_feature, crowd_answer, training=None):
        left_out = self.left_NN(task_feature)
        right_out = self.right_EM(crowd_answer, left_out, type=1)
        return left_out, right_out

class SP_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/SP/"
        self.N_CLASSES = 2
        self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_SP_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.Dropout = Dropout(0.5)

        self.p_pure = 0.125 * np.ones((self.N_CLASSES), dtype=np.float32)
        # self.p = self.p_pure

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))

    def mig_loss_fuction(self, left_out, right_out):

        batch_num = left_out.shape[0]

        I = tf.cast(np.eye(batch_num), dtype=tf.float32)
        E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
        normalize_1 = batch_num
        normalize_2 = batch_num * (batch_num - 1)

        new_output = left_out / self.p_pure
        m = tf.matmul(new_output, right_out, transpose_b=True)
        noise = np.random.rand(1) * 0.0001
        m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
        # m1 = tf.math.log(m * I + E - I)
        m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
        # print(m)

        #loss 来自 KL，与MIG相反数
        return -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2

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
                    row.append(0 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answer_matrix[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)  # task, worker, class
        # print(answers_bin_missings.shape)
        return task_feature, answer_matrix, answers_bin_missings, truth

    def identity_init(self, shape):
        out = np.ones(shape, dtype=np.float32) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return out

    def left_NN(self, input, training=None):
        input = self.bn(input)
        x = self.Dropout(self.fc1(input), training)
        x = self.bn1(x)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return cls_out

    def right_EM(self, crowd_answer, left_out, type=1, training=None):
        crowd_answer = tf.transpose(crowd_answer, (1,0,2))
        # print(crowd_answer)
        crowd_emb = tf.matmul(crowd_answer, self.kernel)
        agg_emb = tf.reduce_sum(crowd_emb, axis=0)
        out = 0
        if type == 1:
            # print(agg_emb.shape)
            # print(left_out.shape)
            # print(self.p_pure.shape)
            out = agg_emb + tf.math.log(left_out+0.001) + tf.math.log(self.p_pure)
        elif type == 2:
            out = agg_emb + tf.math.log(self.p_pure)
        elif type == 3:
            out = agg_emb + tf.math.log(left_out+0.001)
        return tf.nn.softmax(out, axis=-1)

    def call(self, task_feature, crowd_answer, training=None):
        left_out = self.left_NN(task_feature)
        right_out = self.right_EM(crowd_answer, left_out, type=1)
        return left_out, right_out

def run_LableMe():
    batch_size = 64
    trainer = LableMe_model(batch_size)
    train_data, answers, answers_bin_missings, labels_train = trainer.load_LabelMe_dataset()
    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if train_data.shape[0] % batch_size == 0:
        steps = int(train_data.shape[0] / batch_size)
    else:
        steps = int((train_data.shape[0] / batch_size) + 1)
    for epoch in range(60):
        loss = 0
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
            with tf.GradientTape() as tape:
                left_out, right_out = trainer(batch_train_data, batch_answers_bin_missings, training=True)

                loss = trainer.mig_loss_fuction(left_out, right_out)
                # print(loss)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(train_data, answers_bin_missings, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

def run_Music():
    batch_size = 700
    trainer = Music_model(batch_size)
    train_data, answers, answers_bin_missings, labels_train = trainer.load_Music_dataset()
    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if train_data.shape[0] % batch_size == 0:
        steps = int(train_data.shape[0] / batch_size)
    else:
        steps = int((train_data.shape[0] / batch_size) + 1)
    for epoch in range(1000):
        loss = 0
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
            with tf.GradientTape() as tape:
                left_out, right_out = trainer(batch_train_data, batch_answers_bin_missings, training=True)

                loss = trainer.mig_loss_fuction(left_out, right_out)
                # print(loss)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(train_data, answers_bin_missings, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

def run_SP():
    batch_size = 5000
    trainer = SP_model(batch_size)
    train_data, answers, answers_bin_missings, labels_train = trainer.load_SP_dataset()
    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if train_data.shape[0] % batch_size == 0:
        steps = int(train_data.shape[0] / batch_size)
    else:
        steps = int((train_data.shape[0] / batch_size) + 1)
    for epoch in range(1000):
        loss = 0
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
            with tf.GradientTape() as tape:
                left_out, right_out = trainer(batch_train_data, batch_answers_bin_missings, training=True)

                loss = trainer.mig_loss_fuction(left_out, right_out)
                # print(loss)

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(train_data, answers_bin_missings, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

run_LableMe()
# run_Music()
# run_SP()

