# Single Perceptron Visualizer

An interactive web application for visualizing and understanding the behavior of a single perceptron with various activation functions.

## Features

- Interactive visualization of a single perceptron
- Support for multiple activation functions:
  - Step Function
  - Sigmoid
  - ReLU
  - Tanh
  - Softplus
- Real-time calculations and updates
- Adjustable input values and weights
- Network accessibility from other devices on the same network

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/single-perceptron-visualizer.git
cd single-perceptron-visualizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python backend/app.py
```

2. Access the application:
   - On your computer: http://localhost:5000
   - From other devices on your network: http://<your-local-ip>:5000

3. Using the application:
   - Set the number of inputs
   - Choose an activation function
   - Adjust input values and weights
   - View the calculated output in real-time

## Project Structure

```
single-perceptron-visualizer/
├── backend/
│   ├── app.py              # Flask application and perceptron logic
│   └── templates/
│       └── index.html      # Frontend visualization
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

- Flask: Web framework
- NumPy: Numerical computations
- Other dependencies listed in requirements.txt

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask documentation
- NumPy documentation
- Neural Network concepts and perceptron theory 