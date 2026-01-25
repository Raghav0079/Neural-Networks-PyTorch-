# Neural Networks with PyTorch

A comprehensive collection of tutorials and examples demonstrating various PyTorch concepts for neural networks, from basic tensor operations to complete training pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Tutorials](#tutorials)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository provides hands-on examples and tutorials covering the fundamentals of PyTorch for neural network development. It's designed for beginners and intermediate developers who want to learn PyTorch through practical examples.

## 🛠 Prerequisites

- Python 3.7 or higher
- Basic understanding of Python programming
- Familiarity with machine learning concepts (recommended)
- Understanding of linear algebra and calculus (for autograd concepts)

## 📁 Repository Structure

```
Neural-Networks-PyTorch/
├── tensors_in_pytorch.ipynb                               # PyTorch tensor fundamentals
├── pytorch_autograd.ipynb                                 # Automatic differentiation
├── pytorch_nn_module.ipynb                               # Neural network modules
├── pytorch_nn_module.py                                  # NN module Python script
├── dataset_and_dataloader_demo.ipynb                     # Dataset and DataLoader usage
├── dataset_and_dataloader_demo.py                        # Dataset demo Python script
├── pytorch_training_pipeline.py                          # Basic training pipeline
├── pytorch_training_pipeline_using_dataset_and_dataloader.ipynb  # Advanced training with DataLoader
├── pytorch_training_pipeline_using_dataset_and_dataloader.py     # Training pipeline Python script
├── pytorch_training_pipeline_using_nn_module.ipynb       # Training with NN modules
├── pytorch_training_pipeline_using_nn_module.py          # NN module training Python script
└── README.md                                              # This file
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raghav0079/Neural-Networks-PyTorch-.git
   cd Neural-Networks-PyTorch-
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Start with the first tutorial:**
   Open `tensors_in_pytorch.ipynb` to begin your PyTorch journey!

## 📚 Tutorials

### 1. **Tensors in PyTorch** (`tensors_in_pytorch.ipynb`)
- Creating tensors using various methods (`empty`, `zeros`, `ones`, `rand`)
- Tensor operations and manipulations
- GPU/CPU compatibility checks
- Random seed management for reproducibility

### 2. **PyTorch Autograd** (`pytorch_autograd.ipynb`)
- Understanding automatic differentiation
- Computing gradients with `backward()`
- Chain rule applications in neural networks
- Practical examples of gradient computation

### 3. **Neural Network Modules** (`pytorch_nn_module.ipynb`)
- Creating custom neural network classes
- Using `nn.Module` and `nn.Sequential`
- Forward pass implementation
- Activation functions (ReLU, Sigmoid)

### 4. **Dataset and DataLoader Demo** (`dataset_and_dataloader_demo.ipynb`)
- Creating custom datasets with PyTorch Dataset class
- Using DataLoader for batch processing
- Data preprocessing and transformation
- Working with synthetic classification datasets

### 5. **Training Pipelines**
   - **Basic Training Pipeline** (`pytorch_training_pipeline.py`)
     - Complete training workflow
     - Breast cancer dataset example
     - Train/test splitting and data preprocessing
   
   - **Advanced Training with DataLoader** (`pytorch_training_pipeline_using_dataset_and_dataloader.ipynb`)
     - Integrating custom datasets with training loops
     - Batch processing and efficient data loading
   
   - **Training with NN Modules** (`pytorch_training_pipeline_using_nn_module.ipynb`)
     - End-to-end training using custom neural network modules
     - Loss function implementation
     - Optimization strategies

## 💻 Installation

### Using pip:
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib jupyter
```

### Using conda:
```bash
conda install pytorch torchvision -c pytorch
conda install numpy pandas scikit-learn matplotlib jupyter
```

### Verify PyTorch installation:
```python
import torch
print(torch.__version__)
print("GPU available:", torch.cuda.is_available())
```

## 🎮 Usage

### Running Jupyter Notebooks:
```bash
jupyter notebook
```

### Running Python Scripts:
```bash
python pytorch_training_pipeline.py
python pytorch_nn_module.py
python dataset_and_dataloader_demo.py
```

### Example Usage:
```python
import torch
import torch.nn as nn

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = SimpleNN(input_size=10, hidden_size=5, output_size=1)
print(model)
```

## 🔧 Key Features

- **Comprehensive Coverage**: From basic tensors to complete training pipelines
- **Hands-on Examples**: Practical implementations with real datasets
- **Dual Format**: Both Jupyter notebooks and Python scripts provided
- **Beginner Friendly**: Well-commented code with explanations
- **Progressive Learning**: Tutorials build upon each other logically

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- scikit-learn for machine learning utilities
- The open-source community for inspiration and resources

## 📞 Support

If you have any questions or run into issues:
- Open an issue on GitHub
- Check PyTorch documentation: https://pytorch.org/docs/
- Review the examples in each tutorial

---

**Happy Learning with PyTorch! 🎉**

*This repository is actively maintained and updated with new examples and improvements.*
