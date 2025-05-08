from flask import Flask, request, jsonify, render_template
import numpy as np
import socket

app = Flask(__name__)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return 0 if x <= 0.5 else 1

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

def softplus(x):
    return np.log(1 + np.exp(x))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Get form data
            num_inputs = int(request.form.get('num_inputs', 2))
            activation = request.form.get('activation', 'step')
            
            # Get inputs and weights
            inputs = []
            weights = []
            for i in range(num_inputs):
                input_val = float(request.form.get(f'input_{i}', 0))
                weight_val = float(request.form.get(f'weight_{i}', 0.5))
                inputs.append(input_val)
                weights.append(weight_val)
            
            # Convert to numpy arrays
            inputs = np.array(inputs)
            weights = np.array(weights)
            
            # Calculate weighted sum
            weighted_sum = np.dot(inputs, weights)
            
            # Apply activation function
            if activation == 'sigmoid':
                output = sigmoid(weighted_sum)
            elif activation == 'relu':
                output = relu(weighted_sum)
            elif activation == 'tanh':
                output = tanh(weighted_sum)
            elif activation == 'softplus':
                output = softplus(weighted_sum)
            else:  # step function
                output = step_function(weighted_sum)
            
            result = {
                'weighted_sum': float(weighted_sum),
                'output': float(output),
                'activation': activation
            }
        except Exception as e:
            result = {'error': str(e)}
    
    return render_template('index.html', result=result)

def get_local_ip():
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"\nTo access the application:")
    print(f"1. On your computer: http://localhost:5000")
    print(f"2. From other devices on your network: http://{local_ip}:5000")
    print("\nMake sure your firewall allows connections on port 5000")
    app.run(host='0.0.0.0', debug=True, port=5000) 