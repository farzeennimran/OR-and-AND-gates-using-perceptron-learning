import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

#OR gate training data
training_inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_actual_OR = np.array([0, 1, 1, 1])

#AND gate training data
training_inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_actual_AND = np.array([0, 0, 0, 1])

#Training OR gate perceptron
or_gate = Perceptron(input_size=2)
or_gate.train(training_inputs_or, y_actual_OR)

#Training AND gate perceptron
and_gate = Perceptron(input_size=2)
and_gate.train(training_inputs_and, y_actual_AND)

#Testing OR gate
print("OR Gate:")
print("inputs", " ", "y", " ", " y hat")
for inputs, actual_label in zip(training_inputs_or, y_actual_OR):
    predicted = or_gate.predict(inputs)
    print(f"{inputs} -> {actual_label} -> {predicted}")

#Testing AND gate
print("\nAND Gate:")
print("inputs", " ", "y", " ", " y hat")
for inputs, actual_label in zip(training_inputs_and, y_actual_AND):
    predicted = and_gate.predict(inputs)
    print(f"{inputs} -> {actual_label} -> {predicted}")