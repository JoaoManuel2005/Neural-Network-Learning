# Neural Networks

This document provides an overview of neural networks, covering essential terminology, mathematical concepts, and techniques used in both classification and regression tasks. It aims to serve as a reference guide for understanding key neural network concepts.

## Objective
The goal of this README is to explain the fundamentals of neural networks, from basic terminology to advanced mathematical concepts. We will begin with explaining these concepts for a very simple model and build up to more complicated models

## Table of Contents
- [Introduction](#introduction)
- [Terminology](#terminology)
- [General Structure of Neural Networks](#general-structure-of-neural-networks)
- [General Math Concepts](#general-math-concepts)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Optimization](#optimization)
- [Linear Regression](#linear-regression)
- [References](#references)

## Terminology

1. **Classification**: A type of supervised learning where the model predicts categorical labels, such as classifying emails as spam or not spam.
2. **Regression**: A type of supervised learning where the model predicts continuous values, such as predicting the price of a house based on features like size and location.
3. **Training set**: Data used to train the model. Includes input features and output targets
4. **Feature or input variable**: $x$
5. **Target/output variable**: $y$
6. **Number of training examples**: $m$
7. **Single training example**: $(x, y)$
8. **i^th training example**: $(x^{(i)}, y^{(i)})$
9. **y-hat**: $\hat{y}$ - estimate or prediction that model makes for $y$
10. **hypothesis**: function produced by model that takes a new input feature x and outputs a prediction $\hat{y}$
$$
f(x) = \hat{y}
$$
1. **Neuron**: most basic unit in a neural network
2.  **Dense layer**: layer consisting of interconnected neurons, all connected with each other
3.  **Weight**: $w$ - parameter associated with the power (weight) of each connection between neurons. This is a trainable factor of how much of this neurons input to use
4.  **Bias**: $b$ - parameter associated with each neuron. Its purpose is to offset the output positively or negatively, which can further help us map more real-world types of dynamic data
5.  $f_{w,b}(x^{(i)})$: result of model evaluation at $x^{(i)}$ parameterized by $w$, $b$
   
##

## Our first simple model

The most important thing to understand about neural networks is that they are just functions and just like any other functions they simply map a domain $x$ to a range, for the case of neural networks we denote the range as $\hat{y}$:

$$
f(x) = \hat{y}
$$

Let's build our first model, which will simply map the domain $x$ which represents the square meters of a house to a range $\hat{y}$, our prediction of the price of the house. 

The interesting part now is determining what rule (function) we will apply to our domain to map it to our range. Our first neural network will just pass the domain through a single neuron and this neurons output will be our $\hat{y}$.

At this point before we delve into the maths of our function, we'll need to introduce some more terminology:

#### 1. Input Layer
- **Features (Independent Variables)**: The input layer contains neurons that represent the features of your data. These features, denoted as $x$, are the independent variables used by the model to make predictions. For example, in a house price prediction model, features might include the size of the house, its age, and its location.
- The number of neurons in the input layer corresponds to the number of input features.

#### 2. Output Layer
- **Targets (Dependent Variables)**: The output layer consists of neurons that represent the model's predictions, known as the target variables or dependent variables. The number of neurons in the output layer depends on the task:
  - **Single-Output Regression**: One neuron that predicts a single continuous value (e.g., house price).
  - **Multi-Output Regression**: Multiple neurons, each predicting a different continuous value (e.g., house price, buyer age, and tax bracket).
  - **Classification**: The output layer typically uses an activation function like softmax for multi-class classification tasks, where each neuron corresponds to a class.

#### 3. Weights and Biases
- **Weights**: Weights are the parameters associated with the connections between neurons. They determine the importance of the inputs to each neuron and are learned during training.
- **Biases**: Biases are additional parameters that allow the model to shift the output of a neuron. Like weights, biases are also learned during training.

##
Getting back to our model, it will have an input layer which consists of the square meters of a house, then this will be passed through the single neuron we have in the output layer. Since we have one neuron, we will have one weight and one bias associated with this connection. The output of this neuron will be the price of the house. 

So our hypothesis (term for our function $f$) becomes:
$$
f_{w,b}(x) = w * x + b
$$

The next mathematical problem that neural networks is, well, how do we choose the values for $w$ and $b$?

The beauty of neural networks is that we use two algorithms called "forward pass" and "backward pass" choose these values for us in a way that maximizes the times our hypothesis $f(x)$ is correct. 

We will begin with a more high level understanding of these algorithms before we derive a full mathematical understanding of them. 

Durig the forward pass algorithm we take a dataset, called the training data and run it through the network, with the objective of finding out how wrong our current values for $w$ and $b$ are. 
What the backward pass algorithm is that tweaks the values of $w$ and $b$, based on our quantification of how wrong our hypothesis is, with the objective of making our hypothesis less wrong. 

This cycle of quantifying how wrong our hypothesis and then tweaking $w$ and $b$ to make the hypothesis less wrong is repeated as many times as we want until we are happy that our hypothesis is mapping the domain $x$ to the range $\hat{y}$ well. 

Let's consider how we would do this for our example. First we need a dataset which has real true data about the price of houses and their respective size in square meters. This is our training dataset. 
Once we have this we can initialise values for $w$ and $b$. At the moment we won't worry too much about what we initialise them, you can choose any random values. 
These random values form our initial hypothesis. We believe that if you take the size of a house in square meters $x$ and multiply this by a value $w$ called a weight and to the result add a bias $b$ we will get the price of the house. 
Lucky for us we have a dataset of real houses with their size $x$ and their true price $y$. So for the forward pass we will take the size of these houses and see what our current hypothesis is for their price. We can quantify how wrong our hypothesis is by comparing the values that our model predicted for the price of the house $\hat{y}$ with the actual true prices of these houses $y$. 
Then all thats left to do is tweak $w$ and $b$ based on how wrong our current hypothesis is and pray (by pray we mean use some cool backward pass math to ensure) that we have improved it.
If we do this over and over again we should arrive at a hypothesis that predicts the price of a house based on its size. 

If this seems simple, it is because it is. With just one neuron, we have a very simple neural network that is nonetheless powerful and can accurately learn any linear relationship. 

However, as we all know, not many things in life are so linear as the size of a house vs its price. Even this relationship isn't always linear. In many cases we can simplify it to a linear relationship but imagine you are a very large bank and you need a model to help you determine the prices of the many properties you own. You would likely want a model that more accurately captures this relationship without just simplifying it to a linear one as a lot of money is involved. 

So with our new found high level understanding of a simple neural network, we will grow this network by introducing a general structure for it with which it can learn virtually anything, no matter how complex the relationship. 

## General Structure of Neural Networks

Neural networks are composed of layers of interconnected neurons that transform input data (features) into output predictions (targets). The basic structure of a neural network can be divided into the following main components:

### 1. Input Layer
- **Features (Independent Variables)**: The input layer contains neurons that represent the features of your data. These features, denoted as $x$, are the independent variables used by the model to make predictions. For example, in a house price prediction model, features might include the size of the house, its age, and its location.
- The number of neurons in the input layer corresponds to the number of input features.

### 2. Hidden Layers
- **Transformation**: The hidden layers are where the actual "learning" happens. These layers apply transformations to the input features through a series of weighted connections and activation functions.
- **Neurons**: Each neuron in a hidden layer takes inputs from the previous layer, applies weights to these inputs, adds a bias term, and passes the result through an activation function. The output of each neuron is then passed to the next layer.
- **Depth and Width**: The depth of the network refers to the number of hidden layers, while the width refers to the number of neurons in each layer. More layers and neurons allow the network to learn more complex patterns, but they also increase the risk of overfitting if not managed properly.

### 3. Activation Functions
- **Non-linearity**: Activation functions introduce non-linearity into the network, allowing it to learn complex relationships between the input features and the target variables. Common activation functions include the sigmoid function, ReLU (Rectified Linear Unit), and tanh.
- **Role in Hidden Layers**: After each neuron in a hidden layer computes its weighted sum of inputs, the activation function is applied to this sum to determine the neuron's output. This output is then passed on to the next layer.

### 4. Output Layer
- **Targets (Dependent Variables)**: The output layer consists of neurons that represent the model's predictions, known as the target variables or dependent variables. The number of neurons in the output layer depends on the task:
  - **Single-Output Regression**: One neuron that predicts a single continuous value (e.g., house price).
  - **Multi-Output Regression**: Multiple neurons, each predicting a different continuous value (e.g., house price, buyer age, and tax bracket).
  - **Classification**: The output layer typically uses an activation function like softmax for multi-class classification tasks, where each neuron corresponds to a class.

### 5. Weights and Biases
- **Weights**: Weights are the parameters associated with the connections between neurons. They determine the importance of the inputs to each neuron and are learned during training.
- **Biases**: Biases are additional parameters that allow the model to shift the output of a neuron. Like weights, biases are also learned during training.

### 6. Loss Function
- **Purpose**: The loss function measures how far the model's predictions are from the actual target values. It quantifies the "error" of the model and is used to update the weights and biases during training.
- **Examples**: Common loss functions include mean squared error (MSE) for regression tasks and categorical cross-entropy for classification tasks.

### 7. Training with Backpropagation and Gradient Descent
- **Backpropagation**: During training, the model uses backpropagation to calculate the gradients of the loss function with respect to each weight and bias in the network. These gradients indicate how much each parameter needs to change to reduce the loss.
- **Gradient Descent**: The model updates the weights and biases using gradient descent, an optimization algorithm that iteratively adjusts the parameters to minimize the loss function.

### 8. Example of a Neural Network Structure:
- **Input Layer**: 3 neurons representing the features (e.g., size, age, location).
- **Hidden Layers**: 2 hidden layers, each with 4 neurons, using the ReLU activation function.
- **Output Layer**: 1 neuron for single-output regression (e.g., predicting house price).

For multi-output regression, the output layer might have 3 neurons, predicting e.g. house price, buyer age, and buyer tax bracket simultaneously.

This general structure can be applied to various types of tasks, including both regression and classification, by appropriately configuring the output layer and loss functions.

But for the purpose of this README, to explore the maths of this general structure let's choose a multi-output regression model with:
- **Input Layer**: 3 neurons representing the features (e.g., size, age, location).
- **Hidden Layers**: 2 hidden layers, each with 16 neurons, using the ReLU activation function.
- **Output Layer**: 2 neurons for multi-output regression (e.g., predicting house price, buyer age).


##

## General math concepts for understanding neural networks

To understand how neural networks work, it’s essential to grasp the fundamental mathematical operations that occur within a neuron, across hidden layers, and through activation and loss functions. Below is an overview of these concepts before we look at deriving our multi-output regression model:

### 1. Math Inside a Neuron

Each neuron in a neural network performs a simple mathematical operation. It takes a set of input values, applies weights to them, adds a bias, and then passes the result through an activation function.

- **Weighted Sum**: The neuron first computes a weighted sum of the inputs:
  
  $$
  z = w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n + b
  $$

  where:
  - \( x_1, x_2, \dots, x_n \) are the input values (features).
  - \( w_1, w_2, \dots, w_n \) are the weights associated with each input.
  - \( b \) is the bias term.
  - \( z \) is the weighted sum (also called the pre-activation value).

- **Activation Function**: The weighted sum \( z \) is then passed through an activation function to produce the output of the neuron:

  $$
  a = \text{activation}(z)
  $$

  where \( a \) is the activated output of the neuron.

### 2. Math in Hidden Layers

A hidden layer consists of multiple neurons, each performing the operations described above. The output of each neuron in one layer becomes the input to each neuron in the next layer.

- **For each neuron in a hidden layer**:
  
  $$
  z_j^{(l)} = \sum_{i=1}^n w_{ji}^{(l)} \cdot a_i^{(l-1)} + b_j^{(l)}
  $$
  
  where:
  - \( z_j^{(l)} \) is the pre-activation value of neuron \( j \) in layer \( l \).
  - \( w_{ji}^{(l)} \) is the weight from neuron \( i \) in layer \( l-1 \) to neuron \( j \) in layer \( l \).
  - \( a_i^{(l-1)} \) is the activated output from neuron \( i \) in the previous layer.
  - \( b_j^{(l)} \) is the bias term for neuron \( j \) in layer \( l \).

- **Activation**: The pre-activation value \( z_j^{(l)} \) is passed through an activation function to produce the final output of the neuron:

  $$
  a_j^{(l)} = \text{activation}(z_j^{(l)})
  $$

- **Output of the Hidden Layer**: The output of the entire hidden layer can be represented as a function \( h^{(l)} \), which describes the layer’s output with respect to its inputs:

  $$
  h^{(l)}(a^{(l-1)}) = g(W^{(l)} \cdot a^{(l-1)} + b^{(l)})
  $$

  where:
  - \( h^{(l)}(a^{(l-1)}) \) is the output vector of the hidden layer \( l \).
  - \( W^{(l)} \) is the weight matrix for layer \( l \), where each element \( w_{ji}^{(l)} \) represents the weight from neuron \( i \) in layer \( l-1 \) to neuron \( j \) in layer \( l \).
  - \( a^{(l-1)} \) is the vector of activated outputs from the previous layer (layer \( l-1 \)).
  - \( b^{(l)} \) is the bias vector for layer \( l \).
  - \( g(\cdot) \) is the activation function applied element-wise to the vector \( z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)} \).

This function \( h^{(l)}(a^{(l-1)}) \) represents the transformation applied by the hidden layer to the inputs from the previous layer, producing the output that is passed on to the next layer.

### 3. Activation Function

An activation function determines the output of a neuron given its input (the weighted sum plus bias). The activation function introduces non-linearity into the network, enabling it to learn complex patterns that go beyond linear relationships.

- **General Form of an Activation Function**:

  Given the pre-activation value \( z \), the activation function \( g(z) \) produces the activated output \( a \):
  $$
  a = g(z) \\
  \text{where } z = w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n + b
  $$
  

The choice of activation function depends on the specific problem and the architecture of the neural network.

### 4. Loss Function

The loss function measures how well the neural network’s predictions match the actual target values. It quantifies the error of the model, which is then minimized during training through optimization techniques like gradient descent. The loss function takes as its input the final output of the network, i.e. the values of the neurons of the output layer, and the targets. 

- **General Form of a Loss Function**:
  
  For a single training example:
  $$
  L(\hat{y}, y) = \text{some function of } (\hat{y}, y)
  $$
  
  To compute the loss for the entire network across all training examples, we take the average loss over all \( m \) examples in the training set:
  $$
  J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})
  $$

  where:
    - \( J(\theta) \) is the total loss (also known as the cost function).
    - \( \theta \) represents all the parameters of the network, including weights and biases.
    - \( m \) is the number of training examples.
    - \( \hat{y}^{(i)} \) is the predicted value for the \( i \)-th training example.
    - \( y^{(i)} \) is the true value for the \( i \)-th training example.
    - \( L(\hat{y}^{(i)}, y^{(i)}) \) is the loss for the \( i \)-th training example. 
    $\text{}$

  This overall loss \( J(\theta) \) provides a single scalar value that represents how well the network is performing across the entire dataset. The goal during training is to minimize this loss function by adjusting the parameters \( \theta \)


    

### 5. Combining It All: Forward Pass

During a forward pass through the network:
1. The input features are passed through the input layer.
2. Each hidden layer transforms the data using the weighted sums, biases, and activation functions.
3. The final layer produces the network’s output.
4. The loss function is calculated based on the difference between the predicted output and the true target values.

This process is repeated for each training example, and the model parameters (weights and biases) are adjusted to minimize the loss function over time.

### 6. Backward Pass

This is the way with which after each foward pass we adjust the model parameters to minimize the loss function over time. We will go into more detail about this later in this document as an explanation of this requires further mathematical derivations. But first let's visualise a the flow of the forward pass from input to output with the math we have already derived so far.

## Neural Network Flow: From Input to Output

In a neural network, the input \( x \) is first fed into the initial hidden layer, and then the output of each hidden layer is used as the input for the next hidden layer. This process continues until the output layer produces the final prediction. Here's a breakdown of the process:

### 1. Input to the First Hidden Layer
- The input features \( x \) are fed into the first hidden layer.
- The first hidden layer computes \( h^{(1)}(x) \), which is the output of the first hidden layer.

### 2. Processing Through Subsequent Hidden Layers
- The output of the first hidden layer, \( h^{(1)}(x) \), becomes the input to the second hidden layer.
- The second hidden layer computes \( h^{(2)}(h^{(1)}(x)) \), which is the output of the second hidden layer.
- This process continues for all subsequent hidden layers.

### 3. Final Output Layer
- The output of the last hidden layer, \( h^{(n)}(h^{(n-1)}(...h^{(1)}(x)...)) \), becomes the input to the output layer.
- The output layer then computes the final prediction \( \hat{y} \).

### Mathematical Representation

- **First Hidden Layer**:
  $$
  h^{(1)}(x) = g^{(1)}(W^{(1)} \cdot x + b^{(1)})
  $$
  where \( g^{(1)} \) is the activation function for the first hidden layer, \( W^{(1)} \) is the weight matrix, and \( b^{(1)} \) is the bias vector.

- **Second Hidden Layer**:
  $$
  h^{(2)}(h^{(1)}(x)) = g^{(2)}(W^{(2)} \cdot h^{(1)}(x) + b^{(2)})
  $$
  where \( g^{(2)} \), \( W^{(2)} \), and \( b^{(2)} \) are the corresponding activation function, weights, and biases for the second hidden layer.

- **Final Output Layer**:
  $$
  \hat{y} = f_{\theta}(h^{(n)}(x)) = W^{output} \cdot h^{(n)}(x) + b^{output}
  $$
  where \( f_{\theta} \) is result of model evaluation parameterized by all parameters \( \theta \), \( h^{(n)}(x) \) is the output of the last hidden layer, and \( W^{output} \) and \( b^{output} \) are the weights matrix and bias vector for the output layer.

- **Loss calculation**:
  Then to find out how far off our current predictions are, we compare the models predictions with the targets, using:
  $$
  J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})
  $$

### Summary

The input \( x \) is progressively transformed by each hidden layer, with the final transformation producing the output \( \hat{y} \). This sequential process allows the network to learn complex mappings from the input features to the target outputs.

### Applying this to our example

Now that we have derived all this math, let's derive the function for the forward pass algorithm for our example of predicting the price of a house and the age of its buyers

So 

Let's think about the sizes of our layers and their components.
Our input layer has three input features and is connected to the first hidden layer which has 16 neurons. As we said in our definition of the hidden layer, each neuron in the hidden layer takes the whole output of the previous layer as its input. So each of the 16 neurons takes as its input the 3 input features. And for each neuron we have a set of tunable parameters called weights, one for each connection with its inputs. 
Let's consider the nth neuron of the first hidden layer. It has a set of weights 
$$
\begin{bmatrix}
          w_{n,1},
          w_{n,2},
          w_{n,3}
\end{bmatrix}
$$
where $w_{n,1}$ represents the weight the first neuron gives the connection with the first input feature and so on

Each neuron in the hidden layer has this set of tunable weights so we group them as a matrix $W^{(n)}$:
$$
W^{(n)} = 
\begin{bmatrix}
w_{1,1},
w_{1,2},
w_{1,3} \\
w_{2,1},
w_{2,2},
w_{2,3} \\
... \\
w_{15,1},
w_{15,2},
w_{15,3} \\ 
w_{16,1},
w_{16,2},
w_{16,3} \\ 
\end{bmatrix}
$$

Moreover each neuron in the first hidden layer also has a bias term. Each neuron has only one bias term so we group these as a vector $b^{(n)}$:
$$
b^{(n)} = 
\begin{bmatrix}
b_{1} \\
b_{2} \\
... \\
b_{15} \\
b_{16} \\
\end{bmatrix}
$$

So as we saw, $W^{(1)}$ is a (16, 3) matrix and $b^{(1)}$ is a (16, 1) vector

Let's think of the next step of our network which is the flow of information from the first hidden layer to the second hidden layer. This a connection between two layers of 16 neurons. So for each neuron in the second layer we have a tunable weight associated with this neurons connection with each of the previous layer's neurons. So $W^{(2)}$ is a (16, 16) matrix. Just like $b^{(1)}$, $b^{(2)}$ is a (16, 1) vector. 

Next, we have the connection between the second hidden layer and the output layer. The output layer is composed of two neurons, the ouput of one representing the predicted price of the house and the output of the other representing the predicted age of the buyer. So we can determine that $W^{(3)}$ is a (2, 16) matrix and $b^{(3)}$ is a (2, 1) vector. 

So for our model:
$$
\hat{y} = f_{\theta}(h^{(2)}(x)) \\
= W^{3} \cdot h^{(2)}(x) + b^{3} \\
= W^{3} \cdot g^{(2)}(W^{(2)} \cdot h^{(1)}(x) + b^{(2)}) + b^{3} \\
= W^{3} \cdot g^{(2)}(W^{(2)} \cdot g^{(1)}(W^{(1)} \cdot x + b^{(1)}) + b^{(2)}) + b^{3} 
$$
where \( f_{\theta} \) is result of model evaluation parameterized by all parameters \( \theta \), \( h^{(n)}(x) \) is the output of the nth hidden layer and \( g^{(n)}(x) \) is the activation function for the nth hidden layer

Once we have calculated $\hat{y}$ we can determine how wrong this prediction is by comparing it with the actual real values in our training dataset using our loss function: 
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})
$$

Now we get to the real fuckery, the backward pass. 

Unfortunalety as is often in life, before we get to the fuckery, there is more maths. Let's go over the maths we need to understand the backward pass algorithm. And remember that we should frame this exploration of the backpropagation algorithm under the umbrella of the following question:

### How can we correct the error in our network to make better predictions?

First some basic definitions:

#### What is a partial derivative?
$$
f(x,y,z) \rightarrow \frac{\partial}{\partial{x}}f(x,y,z), \frac{\partial}{\partial{y}}f(x,y,z), \frac{\partial}{\partial{z}}f(x,y,z)
$$

Partial derivative of a sum:
$$
f(x,y) = x+y \rightarrow \frac{\partial}{\partial{x}}f(x,y) = \frac{\partial}{\partial{x}}[x+y] = \frac{\partial}{\partial{x}}x +\frac{\partial}{\partial{x}}y = 1 + 0 = 1 \\
\rightarrow \frac{\partial}{\partial{y}}f(x,y) = \frac{\partial}{\partial{y}}[x+y] = \frac{\partial}{\partial{y}}x +\frac{\partial}{\partial{y}}y = 0 + 1 = 1
$$

Partial derivative of multiplication:
$$
f(x,y) = x \cdot y \rightarrow \frac{\partial}{\partial{x}}f(x,y) = \frac{\partial}{\partial{x}}[x \cdot y] = y\frac{\partial}{\partial{x}}x = y \cdot 1 = y \\
\rightarrow \frac{\partial}{\partial{y}}f(x,y) = \frac{\partial}{\partial{x}}[x \cdot y] = x\frac{\partial}{\partial{y}}y = x \cdot 1 = x
$$

#### 
Each of the inputs of a function, have an impact on its output, even if this impact is 0. Thus we represent the gradient of a function as a vector of all of the possible partial derivatives of the function:
$$
    \nabla f(x,y,z) = \begin{bmatrix}
           \frac{\partial}{\partial{x}}f(x,y,z) \\
           \frac{\partial}{\partial{y}}f(x,y,z) \\
           \frac{\partial}{\partial{z}}f(x,y,z)
         \end{bmatrix} = \begin{bmatrix}
           \frac{\partial}{\partial{x}} \\
           \frac{\partial}{\partial{y}} \\
           \frac{\partial}{\partial{z}}
         \end{bmatrix} f(x, y,z)
$$

#### So what is a backward pass and optimization?

The backward pass (also known as backpropagation) is the process of calculating the gradients of the loss function with respect to the network's parameters (weights and biases). This is done by applying the chain rule of calculus, starting from the output layer and moving backward through the network.
Purpose: The purpose of this is to compute how much each parameter in the network contributes to the error (or loss). These gradients are then used to adjust the parameters in a way that reduces the error.

Optimization refers to the process of updating the network's parameters (weights and biases) based on the gradients calculated during the backward pass. The goal of optimization is to minimize the loss function, which measures the difference between the predicted outputs and the actual targets.

The most common optimization algorithm is gradient descent, where parameters are updated by moving them in the direction opposite to the gradient of the loss function (i.e., reducing the loss). But there are various optimization algorithms like Stochastic Gradient Descent (SGD), Adam, RMSprop, etc., which use different strategies for updating the parameters.

#### What is gradient descent?

###### What is the gradient descent algorithm?
In the gradient descent algorithm we update the value of the weights to be the previous value for the weights minus the learning rate multiplied by the partial derivative of the loss function with resepct to the weights:
$$
w = w - \alpha \cdot \frac{\partial}{\partial{w}}J(w,b)
$$
Remember, however, that for multivariate functions, all inputs have an effect on the output. So we should also update the value of the bias terms according to how much the bias terms impact the loss:
$$
b= b - \alpha \cdot \frac{\partial}{\partial{b}}J(w,b)
$$


#### What are the formulas for common activation functions for classification and regression?

The linear activation function:
$$
y = x
$$

The sigmoid activation function:
$$
y = \frac{1}{1-e^{-x}}
$$

The softmax activation function:
$$
S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L}e^{z_{i,l}}}
$$

#### How can we calculate how wrong our model is?

#### What are the formulas for common loss functions for classification?

###### Categorical cross entropy loss: 
This loss function is used for models whose job is to label data with one of many possible classification. These models output layers are a probability distribution, where all of the values represent a confidence level of a given class being the correct class, and where these confidences sum to ​1
$$
L_i = -\sum_jy_{i,j}\log{\hat{y}_{i,j}}
$$
This loss function penalises by the squished nature of the log function as it approaches 0. Since $log(x) \rightarrow - \infty \text{ as } x \rightarrow 0$, when we give a probability $\hat{y}_{ij}$ close to 0 whose true probability $y_{ij}$ is actually a 1, we give this prediction a bad score. The further away our prediction is from 1 (if the target is 1), the more we penalise this prediction.


###### Binary cross entropy loss:
.
$$
L_{i,j} = y_{i,j}\cdot (-\log({-\hat{y}_{i,j}})) + (1-y_{i,j})(-\log({1-\hat{y}_{i,j}})) \\
L_i = \frac{1}{J}\sum_jL_{i,j} \\
L_i = \frac{1}{J}\cdot \sum_j y_{i,j}\cdot (-\log({-\hat{y}_{i,j}})) + (1-y_{i,j})\cdot (-\log({1-\hat{y}_{i,j}}))
$$


#### What are the gradients of common loss functions for classification?

Gradient of categorical cross entropy loss:

$$
L_i = -\sum_jy_{i,j}\log{\hat{y}_{i,j}} \rightarrow \frac{\partial L_i}{\partial \hat{y}_{i,j}} = \frac{\partial}{\partial \hat{y}_{i,j}}[-\sum_jy_{i,j}\log{\hat{y}_{i,j}}] \\
= -\sum_jy_{i,j} \cdot \frac{\partial}{\partial \hat{y}_{i,j}}[\log{\hat{y}_{i,j}}] \\
= -\sum_jy_{i,j} \cdot \frac{1}{\hat{y}_{i,j}} \cdot \frac{\partial}{\partial \hat{y}_{i,j}}\hat{y}_{i,j} \\
= -\sum_jy_{i,j} \cdot \frac{1}{\hat{y}_{i,j}} \cdot 1 \\
= -\sum_j \frac{y_{i,j}}{\hat{y}_{i,j}}  \\
\frac{\partial L_i}{\partial \hat{y}_{i,j}} = \frac{y_{i,j}}{\hat{y}_{i,j}} \\ 
\begin{bmatrix}
                                            \text{since we are calculating the partial derivative with respect to the ​y​ given ​j,} \\
                                            \text {​ the sum is being performed over a single element and can be omitted in the last step}
                                        \end{bmatrix}
$$

Gradient of binary cross entropy loss:


#### What are the gradients of common loss functions for regression?

Gradient for mean squared loss:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^m L_i = \frac{1}{2m} \sum_{i=1}^m \frac{1}{J} \sum_j (y_{i,j} - \hat{y}_{i,j})^2 \\
=\frac{\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^{2}}{2m} \\
\rightarrow
\frac{\partial J(w,b)}{\partial \hat{y}^{(i)}} = 
\frac{\partial}{\partial \hat{y}^{(i)}}[\frac{\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^{2}}{2m}] \\
= \frac{1}{2m} \cdot \frac{\partial}{\partial \hat{y}^{(i)}}[(\hat{y}^{(i)}-y^{(i)})^{2}] \\
= \frac{1}{2m} \cdot  2 \cdot (\hat{y}^{(i)}-y^{(i)}) \frac{\partial}{\partial \hat{y}^{(i)}} [\hat{y}^{(i)}-y^{(i)}] \\
= \frac{1}{2m} \cdot 2 \cdot (\hat{y}^{(i)}-y^{(i)}) \cdot (1-0) \\
= \frac{1}{2m} \cdot  2 \cdot (\hat{y}^{(i)}-y^{(i)})
$$

Gradient for mean absolute error:

$$
J(w,b) = \frac{\sum_{i=1}^{m}\left|(\hat{y}^{(i)}-y^{(i)})\right|}{2m}
\rightarrow
\frac{\partial J(w,b)}{\partial \hat{y}^{(i)}} = 
\frac{\partial}{\partial \hat{y}^{(i)}}[\frac{\sum_{i=1}^{m}\left|(\hat{y}^{(i)}-y^{(i)})\right|}{2m}] \\
= \frac{1}{2m} \cdot \frac{\partial}{\partial \hat{y}^{(i)}}\left|(\hat{y}^{(i)}-y^{(i)})\right| \\
= \frac{1}{2m} \cdot \frac{\partial}{\partial \hat{y}^{(i)}}\left|(\hat{y}^{(i)}-y^{(i)})\right| \\
= \begin{cases}
    \frac{1}{2m} \cdot 1 & \text{if } & \hat{y}^{(i)}-y^{(i)} > 0 \\
    \frac{1}{2m} \cdot -1 & \text{if } & \hat{y}^{(i)}-y^{(i)} < 0
\end{cases}
$$

## Types of Regression in Neural Networks

### Single-Output Regression

#### Overview:
In single-output regression, the model predicts one continuous target variable (dependent variable) based on a set of input features (independent variables).

- **Example**: Predicting the price of a house using features such as size, age, and location.

#### Structure:
- **Input Layer**: Contains neurons corresponding to the input features (e.g., size, age, location).
- **Hidden Layers**: Perform transformations on the inputs through a series of non-linear operations, enabling the model to learn complex patterns.
- **Output Layer**: Contains **one output neuron** that predicts the target variable (e.g., house price).

#### Mathematical Function:
- The function $f_{w,b}(x)$ for single-output regression is:
    $$
    f_{w,b}(x) = w * x + b
    $$
  where $w$ is the weight vector and $b$ is the bias term.

### Multi-Output Regression

#### Overview:
In multi-output regression, the model predicts multiple continuous target variables simultaneously based on the same set of input features.

- **Example**: Predicting the price of a house, the age of potential buyers, and their tax bracket using features such as size, age, and location.

#### Structure:
- **Input Layer**: Contains neurons corresponding to the input features.
- **Hidden Layers**: Similar to single-output regression, these layers perform transformations on the inputs.
- **Output Layer**: Contains **multiple output neurons**, each predicting a different target variable (e.g., house price, buyer age, tax bracket).

#### Mathematical Function:
- The function $f_{W, b}(x)$ for multi-output regression is:
    $$
    f_{W, b}(x) = W * x + b
    $$
  where:
  - $x$ is the input feature vector.
  - $W$ is a weight matrix of dimensions $J \times n$, where $J$ is the number of outputs and $n$ is the number of input features.
  - $b$ is a bias vector of size $J$.


## Math for univariate linear regression model

#### What is the function $f$ ? 
$$
f_{w,b}(x) = wx + b
$$

#### What is the formula for the cost function?
The cost function compares $\hat{y}$ with $y$ and quantifies how wrong the predictions made by the model are by calculating the squared differences. This result is divided by m (the number of training examples) to normalise it or else the cost would increase just because we train on more data, not solely because our predictions on this data are more wrong:

$$
J(w,b) = \frac{\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^{2}}{2m}
= \frac{\sum_{i=1}^{m}(f_{w,b}({x}^{(i)})-y^{(i)})^{2}}{2m}
$$

Our goal is to find values for $w$ and $b$ such that $\hat{y}^{(i)}$ is close to ${y}^{(i)}$ for all $({x}^{(i)}, {y}^{(i)})$ and the cost function helps us quantify at each step how far we are at each stop so we can move closer to the desired outcome

The formula above penalises a prediction more, the further away it is from the target. This penalty increases exponentially. This cost function is called mean squared error.

We can also use a cost function for which this penalty increases linearly:

$$
J(w,b) = \frac{\sum_{i=1}^{m}\left|(\hat{y}^{(i)}-y^{(i)})\right|}{2m}
= \frac{\sum_{i=1}^{m}\left|(f_{w,b}({x}^{(i)})-y^{(i)})\right|}{2m}
$$

This cost function is called mean absolute error.

## Math for multi-output linear regression model

So far we have working with a regression model that outputs one number as its prediction, i.e. it has one output neuron. We use our set of independent variables, the input feaures $x$, to output one dependent variable $\hat{y}$, e.g. using dependent variables - size of house, age of house, location; we have to predict its price
But what if from this set of independent variables we wanted to predict many things?
e.g., from our input features $x$ we wanted to predict not only the price of the house, but also the age of the buyers, and the tax bracket of the buyers so that we can target the right people with ads

Our current model can't do this. For this, we need to define a new function $f$ and new loss function $J(w,b)$:

Suppose we are working with features: size of house, age of house and location; and targets: price of house, age of buyers, and buyers tax bracket. 
In this case:
$$
f_{W, b}(x) = W \cdot x +b
$$
where $x$ is a vector of size 2, $W$ is a matrix of dimensions $3 \times 2$ and $b$ is a vector of size 3


## Conclusion


## References
- [Neural Networks from Scratch by Harrison Kinsley & Daniel Kukieła]()
- [Machine Learning Specialization - Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

