# PyTorch Fundamentals 🔥

A comprehensive collection of PyTorch tutorials and examples covering the fundamentals of deep learning, from basic autograd operations to complete training pipelines.

## 📚 Contents

### 1. [AutoGrad Fundamentals](pytorch_autograd.ipynb)
- **Topics Covered:**
  - Tensor operations and gradient computation
  - Automatic differentiation with `torch.autograd`
  - Gradient tracking and computation graphs
  - Manual gradient calculation examples
  - Backward propagation mechanics

### 2. [Neural Network Modules](pytorch_nn_module.ipynb) | [Python Version](pytorch_nn_module.py)
- **Topics Covered:**
  - Creating custom neural network classes with `nn.Module`
  - Forward pass implementation
  - Sequential model building
  - Model architecture visualization with `torchinfo`
  - Layer initialization and parameter access

### 3. [Complete Training Pipeline](pytorch_training_pipeline.py)
- **Topics Covered:**
  - End-to-end machine learning workflow
  - Data preprocessing and feature engineering
  - Train-test splitting and data normalization
  - Label encoding for classification tasks
  - Converting data to PyTorch tensors
  - **Real-world example**: Breast cancer detection using neural networks

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install torchinfo
pip install pandas numpy scikit-learn
pip install jupyter  # For running notebooks
```

### Quick Start
1. Clone this repository:
   ```bash
   git clone https://github.com/Raghav0079/pytorch-fundamentals.git
   cd pytorch-fundamentals
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # Create this file with the packages above
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Start with `pytorch_autograd.ipynb` to understand the basics!

## 📖 Learning Path

**Recommended order for beginners:**

1. **Start Here** → [AutoGrad Fundamentals](pytorch_autograd.ipynb)
   - Understand how PyTorch computes gradients
   - Learn tensor operations and automatic differentiation

2. **Next** → [Neural Network Modules](pytorch_nn_module.ipynb)
   - Build your first neural network class
   - Understand the `nn.Module` framework

3. **Finally** → [Complete Training Pipeline](pytorch_training_pipeline.py)
   - See everything come together in a real project
   - Learn best practices for ML workflows

## 🎯 Key Concepts Demonstrated

- **Automatic Differentiation**: Understanding how PyTorch tracks and computes gradients
- **Neural Network Architecture**: Building modular, reusable network components
- **Data Pipeline**: From raw data to model-ready tensors
- **Model Training**: Complete workflow including preprocessing, training, and evaluation

## 🛠️ Code Examples

### Simple Neural Network
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Usage
model = SimpleModel(5)
input_tensor = torch.rand(10, 5)
output = model(input_tensor)
```

### Gradient Computation
```python
import torch

# Create tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Compute gradients
y.backward()
print(f"Gradient: {x.grad}")  # dy/dx = 2x + 3 = 7.0
```

## 📊 Project Structure
```
├── pytorch_autograd.ipynb          # Autograd fundamentals
├── pytorch_nn_module.ipynb         # Neural network modules
├── pytorch_nn_module.py            # Python version of NN module
├── pytorch_training_pipeline.py    # Complete ML pipeline
└── README.md                       # This file
```

## 🤝 Contributing

Contributions are welcome! If you have:
- Additional PyTorch examples
- Improvements to existing code
- Better explanations or documentation

Please feel free to open an issue or submit a pull request.

## 📝 Notes

- All notebooks are originally created in Google Colab
- Code includes both interactive notebooks (`.ipynb`) and Python scripts (`.py`)
- Examples use real datasets for practical learning
- Focus on educational clarity over production optimization

## 🔗 Useful Resources

- [Official PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch-book/)

## ⭐ Support

If you find this repository helpful, please consider giving it a star! ⭐

---

**Happy Learning! 🚀**

*Made with ❤️ for the PyTorch community*
