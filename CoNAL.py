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


def CEloss(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, axis=1)
    # print(vec)
    mask = tf.equal(y_true[:, 0, :], -1)
    zer = tf.zeros_like(vec)
    loss = tf.where(mask, x=zer, y=vec)
    return tf.reduce_sum(loss)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/LabelMe/prepared/"
        self.N_CLASSES = 8
        self.data_train_vgg16, self.answers, self.answers_bin_missings, self.labels_train = self.load_LabelMe_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.Dropout = Dropout(0.5)

        self.fc3 = Dense(128, activation=None)
        self.fc4 = Dense(20, activation=None)
        self.fc5 = Dense(20, activation=None)

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
        self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))

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
                    row.append(-1 * np.ones(self.N_CLASSES))
                else:
                    row.append(one_hot(answers[i, r], self.N_CLASSES)[0, :])
            answers_bin_missings.append(row)
        answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # task, class, worker
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
        return out

    def classifier(self, input, training=None):
        # base_model = Sequential()
        # base_model.add(Flatten(input_shape=input.shape[1:]))
        # # base_model.add(Dense(1024, activation='relu'))
        # base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dropout(0.5))
        # base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN
        flatten_input = self.flatten(input)
        x = self.Dropout(self.fc1(flatten_input), training)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return flatten_input, cls_out

    def common_module(self, input):
        instance_difficulty = self.fc3(input)
        instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)

        # instance_difficulty = F.normalize(instance_difficulty)
        worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
        # user_feature = F.normalize(user_feature)
        # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
        common_rate = tf.nn.sigmoid(common_rate)
        return common_rate

    def call(self, input=None, training=None):
        flatten_input, cls_out = self.classifier(input, training)
        common_rate = self.common_module(flatten_input)
        common_prob = tf.matmul(cls_out, self.common_kernel)
        indivi_prob = tf.keras.backend.dot(cls_out, self.kernel)
        crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
        return cls_out, tf.transpose(crowds_out, [0, 2, 1])

class Music_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/music/"
        self.N_CLASSES = 10
        self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_Music_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.Dropout = Dropout(0.5)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.fc3 = Dense(128, activation=None)
        self.fc4 = Dense(80, activation=None)
        self.fc5 = Dense(80, activation=None)

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
        self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))

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

    def classifier(self, input, training=None):
        # base_model = Sequential()
        # base_model.add(Flatten(input_shape=input.shape[1:]))
        # # base_model.add(Dense(1024, activation='relu'))
        # base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dropout(0.5))
        # base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN
        flatten_input = self.flatten(input)
        x = self.bn(flatten_input)
        x = self.Dropout(self.fc1(x), training)
        x = self.bn1(x)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return flatten_input, cls_out

    def common_module(self, input):
        instance_difficulty = self.fc3(input)
        instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)

        # instance_difficulty = F.normalize(instance_difficulty)
        worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
        # user_feature = F.normalize(user_feature)
        # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
        common_rate = tf.nn.sigmoid(common_rate)
        return common_rate

    def call(self, input=None, training=None):
        flatten_input, cls_out = self.classifier(input, training)
        common_rate = self.common_module(flatten_input)
        common_prob = tf.matmul(cls_out, self.common_kernel)
        indivi_prob = tf.keras.backend.dot(cls_out, self.kernel)
        crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
        return cls_out, tf.transpose(crowds_out, [0, 2, 1])

class SP_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_RUNS = 30
        self.DATA_PATH = "./dataset/SP/"
        self.N_CLASSES = 2
        self.train_data, self.answers, self.answers_bin_missings, self.labels_train = self.load_SP_dataset()
        self.N_ANNOT = self.answers.shape[1]
        self.worker_feature = np.eye(self.N_ANNOT)

        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.N_CLASSES)
        self.Dropout = Dropout(0.5)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.fc3 = Dense(128, activation=None)
        self.fc4 = Dense(20, activation=None)
        self.fc5 = Dense(20, activation=None)

        self.kernel = tf.Variable(self.identity_init((self.N_ANNOT, self.N_CLASSES, self.N_CLASSES)))
        self.common_kernel = tf.Variable(self.identity_init((self.N_CLASSES, self.N_CLASSES)))

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

    def classifier(self, input, training=None):
        # base_model = Sequential()
        # base_model.add(Flatten(input_shape=input.shape[1:]))
        # # base_model.add(Dense(1024, activation='relu'))
        # base_model.add(Dense(128, activation='relu'))
        # base_model.add(Dropout(0.5))
        # base_model.add(Dense(self.N_CLASSES))
        # base_model.add(Activation("softmax"))
        # base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #                    loss='categorical_crossentropy')  ## for EM-NN
        flatten_input = self.flatten(input)
        x = self.bn(flatten_input)
        x = self.Dropout(self.fc1(x), training)
        x = self.bn1(x)
        cls_out = tf.nn.softmax(self.fc2(x), axis=-1)
        return flatten_input, cls_out

    def common_module(self, input):
        instance_difficulty = self.fc3(input)
        instance_difficulty = tf.math.l2_normalize(self.fc4(instance_difficulty), axis=-1)

        # instance_difficulty = F.normalize(instance_difficulty)
        worker_feature = tf.math.l2_normalize(self.fc5(self.worker_feature), axis=-1)
        # user_feature = F.normalize(user_feature)
        # common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = tf.matmul(instance_difficulty, worker_feature, transpose_b=True)
        common_rate = tf.nn.sigmoid(common_rate)
        return common_rate

    def call(self, input=None, training=None):
        flatten_input, cls_out = self.classifier(input, training)
        common_rate = self.common_module(flatten_input)
        common_prob = tf.matmul(cls_out, self.common_kernel)
        indivi_prob = tf.keras.backend.dot(cls_out, self.kernel)
        crowds_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob
        return cls_out, tf.transpose(crowds_out, [0, 2, 1])

def run_LableMe():
    trainer = LableMe_model()
    batch_size = 10000
    train_data, answers, answers_bin_missings, labels_train = trainer.load_LabelMe_dataset()
    shuffle_train_data, shuffle_answers_bin_missings = shuffle_data(train_data, answers_bin_missings, batch_size)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if train_data.shape[0] % batch_size == 0:
        steps = int(train_data.shape[0] / batch_size)
    else:
        steps = int((train_data.shape[0] / batch_size) + 1)
    for epoch in range(50):
        loss = 0
        print('epoch:', epoch)
        for step in range(steps):
            # print('step:', step)
            batch_train_data = shuffle_train_data[step * batch_size:(step + 1) * batch_size, :]
            batch_answers_bin_missings = shuffle_answers_bin_missings[step * batch_size:(step + 1) * batch_size, :]
            with tf.GradientTape() as tape:
                _, crowds_out = trainer(input=batch_train_data, training=True)
                # print(crowds_out.shape)
                # print(answers_bin_missings.shape)
                s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)),
                                         axis=-1)
                # print(s)

                loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) - (0.00001 * tf.reduce_sum(s))

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(input=train_data, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

def run_Music():
    trainer = Music_model()
    batch_size = 700
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
                _, crowds_out = trainer(input=batch_train_data, training=True)
                # print(crowds_out.shape)
                # print(answers_bin_missings.shape)
                s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)),
                                         axis=-1)
                # print(s)

                loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) - (0.00001 * tf.reduce_sum(s))

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(input=train_data, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

def run_SP():
    trainer = SP_model()
    batch_size = 5000
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
                _, crowds_out = trainer(input=batch_train_data, training=True)
                # print(crowds_out.shape)
                # print(answers_bin_missings.shape)
                s = tf.math.l2_normalize(tf.reshape(trainer.kernel - trainer.common_kernel, (trainer.N_ANNOT, -1)),
                                         axis=-1)
                # print(s)

                loss = CEloss(y_true=batch_answers_bin_missings, y_pred=crowds_out) - (0.00001 * tf.reduce_sum(s))

                vars = tape.watched_variables()
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

        cls_out, _ = trainer(input=train_data, training=False)

        flag = tf.compat.v1.to_int32(tf.equal(tf.argmax(cls_out, axis=-1), labels_train))
        print('Acc:', tf.reduce_sum(flag) / labels_train.shape[0])
        # print('.................')

# run_Music()
run_SP()