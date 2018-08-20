# Notes


The **keras.models.Sequential** class is a wrapper for the neural network model that treats the network as a sequence of layers. It implements the Keras model interface with common methods like compile(), fit(), and evaluate() that are used to train and run the model. We'll cover these functions soon, but first let's start looking at the layers of the model.

Only the input dimension of the first layer must be explicitly stated. Keras will automatically infer the shape of all other layers.

It is common to explicitly separate the activation layers because it allows direct access to the outputs of each layer before the activation is applied

Example of a model architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 32)                96        
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
_________________________________________________________________
activation_2 (Activation)    (None, 1)                 0         
=================================================================
Total params: 129
Trainable params: 129
Non-trainable params: 0
_________________________________________________________________
```

Optimisers in Keras: [https://keras.io/optimizers/](https://keras.io/optimizers/)

Adam optimizer: 

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
