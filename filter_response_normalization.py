import tensorflow as tf


class FilterResponseNormalization(tf.keras.layers.Layer):
    """
    Filter response normalization layer (Singh S, Krishnan S 2019)
    
    Arguments:
        eps : float, optional
            Epsilon value to avoid division by zero, by default 1e-15
        beta_initializer : str, optional
            Initializer for beta weight, by default "ones"
        beta_regularizer : str, optional
            Regularizer for beta weight, by default None
        beta_constraint : str, optional
            Constraint for beta weight, by default None
        gamma_initializer : str, optional
            Initializer for gamma weight, by default "zeros"
        gamma_regularizer : str, optional
            Regularizer for gamma weight, by default None
        gamma_constraint : str, optional
            Constraint for gamma weight, by default None
        tau_initializer : str, optional
            Initializer for tau weight, by default "zeros"
        tau_regularizer : str, optional
            Regularizer for tau weight, by default None
        tau_constraint : str, optional
            Constraint for tau weight, by default None

    Input shape:
        Any 2D array with arbitrary number of channels, shape HxWxC

    Output shape:
        Same shape as input.

    References:
        - [Filter Response Normalization](https://arxiv.org/abs/1607.06450)
    """

    def __init__(
        self,
        eps=1e-15,
        beta_initializer="ones",
        gamma_initializer="zeros",
        tau_initializer="zeros",
        beta_regularizer=None,
        gamma_regularizer=None,
        tau_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        tau_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)

    def build(self, input_shape):
        """
        Intialize weights, bias and threshold variables
        """

        shape = input_shape[-1:]

        self.beta = self.add_weight(
            shape=shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            name="beta",
        )
        self.gamma = self.add_weight(
            shape=shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            name="gamma",
        )
        self.tau = self.add_weight(
            shape=shape,
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
            name="tau",
        )

    def call(self, x):
        """        
        Arguments:
            x : tensorflow tensor
                Input tensor of shape NxHxWxC
        
        Returns:
            tensorflow tensor
                Output tensor with the filter response normalization and thresholded 
                linear unit activation
        """

        # Computes the mean norm of activations per channel.
        nu2 = tf.math.reduce_mean(tf.math.square(x), axis=[1, 2], keepdims=True)
        # Performs FRN: x = x/sqrt(nu^2)
        x = x * tf.math.rsqrt(nu2 + self.eps)
        # Performs TLU activation
        x = tf.math.maximum(self.beta * x + self.gamma, self.tau)
        return x

    def compute_output_shape(input_shape):
        """
        The output shape equals the input shape.
        """
        return input_shape

    def get_config(self):
        config = {
            "epsilon": self.eps,
            "beta_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
