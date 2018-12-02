
######################
# the learning lab
# original code from https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
######################

import operator
import numpy as np
import sys
from datetime import datetime

################################# Input and Set ################################
vocabulary_size = 300
filename = 'try_text.csv'

################################# Building the RNN #################################
def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores))

def predct(y_out):
    return [index_to_word[i] for i in np.argmax(y_out, axis=1)]


class Numpy_RNN():
    def __init__(self, vab_dim, hidden_dim=20, bptt_truncate=4):
        ##################
        # U: W_xh  W: W_hh  V: W_hy
        ##################
        # Assign instance variables
        self.vab_dim = vab_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Randomly initialize the network parameters
        # self.W = np.concatenate((W, U), axis=1)
        self.U = np.random.uniform(-np.sqrt(1. / vab_dim), np.sqrt(1. / vab_dim), (hidden_dim, vab_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (vab_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        ##################
        # h[t] = U*x + W*h[t-1]
        # y_out = V*h[t]
        ##################
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        y_out = np.zeros((T, self.vab_dim))
        print(y_out.shape)
        for t in range(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            h[t] = np.tanh(self.U[:, x[t]] + self.W.dot(h[t - 1]))
            y_out[t] = softmax(self.V.dot(h[t]))
        return h, y_out # (59,20), (58, 300);


    def loss(self, X, Y):
        ##################
        # X is 2d-array
        # Y is the same dimension as X
        ##################
        L = 0
        # n_sents = len(Y)
        # n_words/T = len(Y[i])
        for i in range(len(Y)):
            h, y_out = self.forward_propagation(X[i])
            correct_word_preds = y_out[np.arange(len(Y[i])), Y[i]]
            L += -1*np.sum(np.log(correct_word_preds))

        Num = np.sum((len(yi) for yi in Y))
        return L/Num

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        y_out, h = self.forward_propagation(x)

        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        # delta_y = y_out-y_true
        delta_y = y_out
        delta_y[np.arange(len(y)), y] -= 1.

        # For each output backwards...
        for t in range(T)[::-1]:
            dLdV += np.outer(delta_y[t], h[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_y[t]) * (1 - (h[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in range(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print("Backpropagation step t=%d bptt step=%d " % (t, bptt_step))
                dLdW += np.outer(delta_t, h[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - h[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    # Whenever you implement backpropagation it is good idea to also implement gradient checking,
    # which is a way of verifying that your implementation is correct.
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                            np.abs(backprop_gradient) + np.abs(estimated_gradient))
                it.iternext()

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


    def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

################################# Main #################################
## try forward_propagation
np.random.seed(10)
x = X_tr[10]
model = Numpy_RNN(300)
h, yo = model.forward_propagation(x)
print(yo.shape)
predct(yo)

############### try
U = np.arange(20).reshape((2,10))
x = np.arange(10)
h = np.arange(2)
x[h]
U.dot(x)
U[:,x]
