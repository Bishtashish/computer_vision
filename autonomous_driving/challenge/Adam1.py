import numpy as np
# SOURCE: https://www.deeplearning.ai/deep-learning-specialization/
def initialize_adam(parameters,num_layers) :
    """
    Initializes m and v as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    m -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    v -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """
    L = len(parameters) // num_layers # number of layers in the neural networks
    m = {} # first moment vector
    v = {} # second moment vector
    # Initialize m, v. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        m["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        m["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return m, v
def update_parameters_with_adam(parameters,num_layers, grads, m, v, t, learning_rate = 0.01,
    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    m -- Adam variable, moving average of the first gradient, python dictionary
    v -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    Returns:
    parameters -- python dictionary containing your updated parameters 
    m -- Adam variable, moving average of the first gradient, python dictionary
    v -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // num_layers        # number of layers in the neural networks
    m_corrected = {}                         # Initializing first moment estimate, python dictionary
    v_corrected = {}                         # Initializing second moment estimate, python dictionary
    # Perform Adam update on all parameters
    for l in range(L):
    # Moving average of the gradients. Inputs: "m, grads, beta1". Output: "m".
        m["dW" + str(l+1)] = beta1 * m["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        m["db" + str(l+1)] = beta1 * m["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        # Compute bias-corrected first moment estimate. Inputs: "m, beta1, t". Output: "m_corrected".
        m_corrected["dW" + str(l+1)] = m["dW" + str(l+1)] / (1 - np.power(beta1,t))
        m_corrected["db" + str(l+1)] = m["db" + str(l+1)] / (1 - np.power(beta1,t))
        # Moving average of the squared gradients. Inputs: "v, grads, beta2". Output: "v".
        v["dW" + str(l+1)] = beta2 * v["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)],2)
        v["db" + str(l+1)] = beta2 * v["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)],2)
        # Compute bias-corrected second raw moment estimate. Inputs: "v, beta2, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta2,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta2,t))
        # Update parameters. Inputs: "parameters, learning_rate, m_corrected, v_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * m_corrected["dW" + str(l+1)] / (np.sqrt(v_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * m_corrected["db" + str(l+1)] / (np.sqrt(v_corrected["db" + str(l+1)]) + epsilon)
    return parameters, m, v