import tensorflow as tf


class FilterResponseNormalization(tf.keras.layers.Layer):
    """
    Filter response normalization layer according to https://arxiv.org/pdf/1911.09737.pdf
    
    Parameters
    ----------
    eps : float, optional
        Epsilon value to avoid division by zero, by default 1e-15
    weight_initializer : str, optional
        Initializer for weights, by default "ones"
    weight_regularizer : [type], optional
        Regularizer for weights, by default None
    weight_constraint : [type], optional
        Constraints for weights, by default None
    bias_initializer : str, optional
        Initializer for biases, by default "zeros"
    bias_regularizer : [type], optional
        Regularizer for biases, by default None
    bias_constraint : [type], optional
        Constraints for biases, by default None
    threshold_initializer : str, optional
        Initializer for thresholded unit, by default "zeros"
    threshold_regularizer : [type], optional
        Regularizer for thresholded unit, by default None
    threshold_constraint : [type], optional
        Constraints for thresholded unit, by default None
    """

    def __init__(
        self,
        eps=1e-15,
        weight_initializer="ones",
        weight_regularizer=None,
        weight_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        threshold_initializer="zeros",
        threshold_regularizer=None,
        threshold_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = tf.Variable(eps)
        self.weight_initializer = tf.keras.initializers.get(weight_initializer)
        self.weight_regularizer = tf.keras.regularizers.get(weight_regularizer)
        self.weight_constraint = tf.keras.constraints.get(weight_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.threshold_constraint = tf.keras.constraints.get(threshold_constraint)
        self.threshold_regularizer = tf.keras.regularizers.get(threshold_regularizer)
        self.threshold_initializer = tf.keras.initializers.get(threshold_initializer)

    def build(self, input_shape):
        """
        Intialize weights, bias and threshold variables
        """

        shape = input_shape[-1:]
        self.beta = self.add_weight(
            shape=shape,
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            constraint=self.weight_constraint,
            name="beta",
        )
        self.gamma = self.add_weight(
            shape=shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name="gamma",
        )
        self.tau = self.add_weight(
            shape=shape,
            initializer=self.threshold_initializer,
            regularizer=self.threshold_regularizer,
            constraint=self.threshold_constraint,
            name="tau",
        )
        super().build(input_shape)

    def call(self, x):
        """        
        Parameters
        ----------
        x : tensorflow tensor
            Input tensor of shape NxHxWxC
        
        Returns
        -------
        tensorflow tensor
            Output tensor with the filter response normalization and thresholded linear unit activation
        """

        # Compute the mean norm of activations per channel.
        nu2 = tf.math.reduce_mean(tf.math.square(x), axis=[1, 2], keepdims=True)
        # Perform FRN: x = x/sqrt(nu^2)
        x = x * tf.math.rsqrt(nu2 + tf.math.abs(self.eps))
        # Perform TLU activation
        x = tf.math.maximum(self.beta * x + self.gamma, self.tau)
        return x

    @staticmethod
    def compute_output_shape(input_shape):
        """
        The output shape will be the same as the input shape.
        """
        return input_shape

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "beta_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.weight_initializer
            ),
            "tau_initializer": tf.keras.initializers.serialize(
                self.threshold_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.weight_regularizer
            ),
            "tau_regularizer": tf.keras.regularizers.serialize(
                self.threshold_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.weight_constraint),
            "tau_constraint": tf.keras.constraints.serialize(self.threshold_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
