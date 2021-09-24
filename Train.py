import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork



def make_predictions(a_out):
    return np.argmax(a_out, 0)

def find_accuracy(predictions, outputs):
    return np.sum(predictions == outputs) / outputs.size

train = pd.read_csv("sign_mnist_train.csv")
train = np.array(train)
np.random.shuffle(train)

train = train.T

train_outs = train[0]
train_ins = train[1:len(train)]
train_ins = train_ins/255

test = pd.read_csv("sign_mnist_test.csv")
test = np.array(test)
np.random.shuffle(test)

test = test.T

test_outs = test[0]
test_ins = test[1:len(test)]
test_ins = test_ins/255

n=NeuralNetwork(784, 350, 26)

accuracy_test = 0
i = 0
accuracy_required = 0.70
accuracy_level = 1

while accuracy_required < 0.96:

    while accuracy_test < accuracy_required:
        i+=1
        n.train(train_ins, train_outs)
        predictions_train = make_predictions(n.predict(train_ins))
        predictions_test = make_predictions(n.predict(test_ins))
        accuracy_train = find_accuracy(predictions_train, train_outs)
        accuracy_test = find_accuracy(predictions_test, test_outs)
        print("Training iteration:", i)    
        print("Training accuracy:", accuracy_train)
        print("Testing accuracy:", accuracy_test)


    np.savetxt("weights-and-biases/hidden_weights"+str(accuracy_level)+".csv", n.hidden_weights, delimiter=",", fmt="%s")
    np.savetxt("weights-and-biases/hidden_bias"+str(accuracy_level)+".csv", n.hidden_bias, delimiter=",", fmt="%s")
    np.savetxt("weights-and-biases/output_weights"+str(accuracy_level)+".csv", n.output_weights, delimiter=",", fmt="%s")
    np.savetxt("weights-and-biases/output_bias"+str(accuracy_level)+".csv", n.output_bias, delimiter=",", fmt="%s")

    accuracy_required += 0.05
    accuracy_level += 1
