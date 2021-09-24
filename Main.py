import cv2
from random import random as r
import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork

def make_predictions(a_out):
    return np.argmax(a_out, 0)

def find_accuracy(predictions, outputs):
    return np.sum(predictions == outputs) / outputs.size


hidden_w = pd.read_csv("weights-and-biases/hidden_weights1.csv", header=None)
hidden_b = pd.read_csv("weights-and-biases/hidden_bias1.csv", header=None)
output_w = pd.read_csv("weights-and-biases/output_weights1.csv", header=None)
output_b = pd.read_csv("weights-and-biases/output_bias1.csv", header=None)

hidden_w = np.array(hidden_w)
hidden_b = np.array(hidden_b)
output_w = np.array(output_w)
output_b = np.array(output_b)


n = NeuralNetwork(784, 350, 26)

n.set_weights_and_biases(hidden_w, hidden_b, output_w, output_b)

cam = cv2.VideoCapture(1)

cv2.namedWindow("Interpreter")

def num_to_char(num):
    alphabet = {
        0: "a",
        1: "b",
        2: "c",
        3: "d",
        4: "e",
        5: "f",
        6: "g",
        7: "h",
        8: "i",
        9: "j",
        10: "k",
        11: "l",
        12: "m",
        13: "n",
        14: "o",
        15: "p",
        16: "q",
        17: "r",
        18: "s",
        19: "t",
        20: "u",
        21: "v",
        22: "w",
        23: "x",
        24: "y",
        25: "z",
    }

    return alphabet[num]

print("Press escape to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("test", frame)

    frame = cv2.resize(frame, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.ravel().reshape(784, 1)
    frame = frame/255

    prediction = make_predictions(n.predict(frame))
    prediction = num_to_char(int(prediction[0]))

    print(prediction)

    k = cv2.waitKey(1)
    if k%256 == 27:
        break

cam.release()
cv2.destroyAllWindows()