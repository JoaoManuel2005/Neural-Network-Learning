# Neural Networks

## Terminology

1. **Classification**:
2. **Regression**: 
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

## General math concepts essential to neural networks for classification and regression

#### What is a neuron?

#### What is an activation functiom?

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



#### How can we correct the error in our network?



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

#### What is a gradient of a function in terms of partial derivatives?
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

#### What is optimization?

#### What is gradient descent?

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
J(w,b) = \frac{\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^{2}}{2m} 
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
= \frac{1}{2m} \cdot 1 \text{ if  } \hat{y}^{(i)}-y^{(i)} > 0 \\
\frac{1}{2m} \cdot -1 \text{  if  } \hat{y}^{(i)}-y^{(i)} < 0
$$

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


