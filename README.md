# OR and AND Gates using Perceptron Learning

This repository demonstrates how to use a single-layer perceptron to simulate the behavior of OR and AND gates. A perceptron is a type of artificial neural network that can be used for binary classification tasks. In this project, we will train perceptrons to recognize the outputs of OR and AND logic gates.

## Introduction

A perceptron is one of the simplest types of artificial neural networks. It consists of one or more input nodes, a set of weights, and a single output node. The perceptron makes predictions by calculating a weighted sum of the inputs and applying an activation function to this sum.

## Neural Networks and Perceptron Learning

Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process input data to make predictions or decisions. The perceptron is a basic building block of neural networks and can be used to solve linearly separable problems.

The perceptron learning algorithm adjusts the weights based on the error between the predicted output and the actual output. This process is repeated over multiple iterations (epochs) until the perceptron learns to correctly classify the training data.

## What, Why, and How

- **What**: This project demonstrates how a perceptron can learn to perform the logical operations of OR and AND gates.
- **Why**: Understanding the perceptron learning algorithm is fundamental to learning more complex neural network models.
- **How**: By training a perceptron on the input-output pairs of OR and AND gates, we can see how the perceptron adjusts its weights to make accurate predictions.

## Code Explanation

The code in this repository is written in Python and uses the NumPy library for numerical operations. Here is a step-by-step explanation of what the code does:

1. **Define the Perceptron Class**: The `Perceptron` class has methods for initializing the perceptron, making predictions, and training the perceptron.
    ```python
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
    ```

2. **Prepare Training Data**: We define the training data for OR and AND gates.
    ```python
    training_inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_actual_OR = np.array([0, 1, 1, 1])

    training_inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_actual_AND = np.array([0, 0, 0, 1])
    ```

3. **Train the Perceptrons**: We create and train perceptrons for OR and AND gates.
    ```python
    # Training OR gate perceptron
    or_gate = Perceptron(input_size=2)
    or_gate.train(training_inputs_or, y_actual_OR)

    # Training AND gate perceptron
    and_gate = Perceptron(input_size=2)
    and_gate.train(training_inputs_and, y_actual_AND)
    ```

4. **Test the Perceptrons**: We test the trained perceptrons to verify their predictions.
    ```python
    # Testing OR gate
    print("OR Gate:")
    for inputs, actual_label in zip(training_inputs_or, y_actual_OR):
        predicted = or_gate.predict(inputs)
        print(f"{inputs} -> {actual_label} -> {predicted}")

    # Testing AND gate
    print("\nAND Gate:")
    for inputs, actual_label in zip(training_inputs_and, y_actual_AND):
        predicted = and_gate.predict(inputs)
        print(f"{inputs} -> {actual_label} -> {predicted}")
    ```

## Results

After training, the perceptrons can correctly predict the outputs of the OR and AND gates:

- **OR Gate**:

inputs    y    y_hat

[0, 0] -> 0 -> 0

[0, 1] -> 1 -> 1

[1, 0] -> 1 -> 1

[1, 1] -> 1 -> 1

- **AND Gate**:

inputs    y    y_hat

[0, 0] -> 0 -> 0

[0, 1] -> 0 -> 0

[1, 0] -> 0 -> 0

[1, 1] -> 1 -> 1


## Conclusion

This project demonstrates the use of the perceptron learning algorithm to model simple logical gates. By understanding how the perceptron works, you can build more complex neural networks and tackle a wider range of machine learning problems.
