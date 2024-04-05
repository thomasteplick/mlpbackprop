<h3>Multilayer Perceptron Neural Network with the Back-Propagation Algorithm</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\MultilayerPerceptron\src\backprop\ directory and issue "go run ann.go" to
start the Multilayer Perceptron (MLP) Neural Network server. In a web browser enter http://127.0.0.1:8080/mlpbackprop
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, examples consisting of x-y coordinates in the Cartesian Plane and the desired class are supplied to the network.
The network itself, is a directed graph consisting of an input layer of nodes, one or more hidden layers of nodes, and
an output layer of nodes.  Each layer of nodes can be arbitrarily deep.  The nodes of the network are connected by weighted
links.  The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  <b>The purpose of this program is to classify an x-y coordinate
in the Cartesian Plane to a class (a number)</b>.  The boundary of each class can be in the shape of a square or a circle.
The user selects the MLP training parameters:
<li>Hidden Layers</li>
<li>Layer Depth</li>
<li>Training Examples</li>
<li>Classes</li>
<li>Learning Rate</li>
<li>Momentum</li>
<li>Epochs</li>
<li>Separation</li>
<li>Ensembles</li>
<li>Class Shape</li>
<br>
<p>
The <i>Learning Rate</i> and <i>Momentum</i> must be less than one.  Each <i>Epoch</i> consists of the number of <i>Training Examples</i>.  
One training example is the x-y coordinate of the point in the Cartesian Plane and the desired class (0, 1, …).
The <i>Separation</i> specifies how far apart the classes are.  Entering zero means there is no distance between the classes.
Each class can therefore be regarded as a cluster of points; the hidden layers determine where those clusters are.
The <i>Ensembles</i> are used to average the Epochs and to minimize the variance of the mean-square-error (MSE).  
It can be just set to one.  The <i>Class Shape</i> is the shape of the boundary of each cluster of points.
</p>
<p>
When the <i>Submit</i> button on the MLP Training Parameters form is clicked, the weights in the network are trained
and the mean-square-error (MSE) is graphed versus Epoch.  As can be seen in the screen shots below, there is significant variance over the ensemble.
One way to reduce the variance is to select more than one ensemble; the ensemble average of the MSE will then be plotted.  However,
the weights are not averaged over the ensembles.  The final weights are the result of one ensemble, the last one.
</p>
<p>
When the <i>Test</i> link is clicked, 10,000 examples are supplied to the MLP.  It classifies the x-y coordinates.
The test results are tabulated and graphed.  Each class has a different color to make it easier to understand the results.
As can be seen in the screen shots below, the farther apart the classes are, the better the results.  
It takes some trial-and-error with the MLP Training Parameters to reduce the MSE to zero.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the MLP.  In general, the more neurons in the
network, the more training examples will be needed to reduce the MSE to zero.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>

![image](https://github.com/thomasteplick/mlpbackprop/assets/117768679/0e70faa5-0cb7-4b97-84b5-4e76cdaaaa22)

<h3>MLP, 25 Classes, Circle Class Shape, Training Phase, MSE vs Epoch</h3>

![MLP25CircleTrain](https://github.com/thomasteplick/mlpbackprop/assets/117768679/9fe9f420-57d9-48e8-a289-d247f827c934)

<h3>MLP, 25 Classes, Circle Class Shape, Testing Phase, Classification Test Results</h3>

![MLP25CircleTest](https://github.com/thomasteplick/mlpbackprop/assets/117768679/38a4f83f-e4c7-4edb-8536-7da84f68d7b6)

<h3>MLP, 25 Classes, Square Class Shape, Training Phase, MSE vs Epoch</h3>

![MLP25SquareTrain](https://github.com/thomasteplick/mlpbackprop/assets/117768679/2651336a-a1f9-42fe-820c-da0458f4483f)

<h3>MLP, 25 Classes, Square Class Shape, Testing Phase, Classification Test Results</h3>

![MLP25SquareTest](https://github.com/thomasteplick/mlpbackprop/assets/117768679/5d73e9f3-27ee-47fc-af58-ec5fe8ef4022)

<h3>MLP, 25 Classes, Circle Class Shape, Training Phase, Separation=1, MSE vs Epoch</h3>

![MLP25CircleTrainSep1](https://github.com/thomasteplick/mlpbackprop/assets/117768679/d8c65d0f-18e3-4c8f-811d-53fe751f0a2b)

<h3>MLP, 25 Classes, Circle Class Shape, Testing Phase, Separation=1, Classification Test Results</h3>

![MLP25CircleTestSep1](https://github.com/thomasteplick/mlpbackprop/assets/117768679/f43d59f1-c680-4746-b038-8d5b226ebb5b)

<h3>MLP, One Class, Circle Class Shape, Testing Phase, Classifcation Test Results</h3>

![MLP1CircleTest](https://github.com/thomasteplick/mlpbackprop/assets/117768679/a0d8f960-0c4f-420b-bf5a-4158462716e5)


