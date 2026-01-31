# Neural Networks PyTorch Implementation Repository

A comprehensive collection of PyTorch tutorials and implementations covering fundamental concepts to advanced neural network techniques. This repository serves as a practical guide for learning PyTorch and building neural networks from scratch.

## 🔥 Features

- **Complete PyTorch Fundamentals**: From tensors to advanced neural architectures
- **Real-world Applications**: Fashion-MNIST classification with various optimization techniques
- **GPU Optimization**: CUDA-enabled implementations for faster training
- **Hyperparameter Tuning**: Optuna integration for automated hyperparameter optimization
- **Best Practices**: Proper dataset handling, training pipelines, and model evaluation

## 📚 Repository Contents

### 🧮 Core PyTorch Concepts
- **[tensors_in_pytorch.ipynb](tensors_in_pytorch.ipynb)** - Introduction to PyTorch tensors, operations, and GPU usage
- **[pytorch_autograd.ipynb](pytorch_autograd.ipynb)** - Automatic differentiation and gradient computation
- **[pytorch_nn_module.ipynb](pytorch_nn_module.ipynb)** - Building neural networks using `nn.Module`

### 🗂️ Data Handling
- **[dataset_and_dataloader_demo.ipynb](dataset_and_dataloader_demo.ipynb)** - Custom Dataset and DataLoader implementation
- **[pytorch_training_pipeline_using_dataset_and_dataloader.ipynb](pytorch_training_pipeline_using_dataset_and_dataloader.ipynb)** - Complete training workflow with proper data handling

### 🏋️ Training Pipelines
- **[pytorch_training_pipeline.ipynb](pytorch_training_pipeline.ipynb)** - Basic training loop implementation
- **[pytorch_training_pipeline_using_nn_module.ipynb](pytorch_training_pipeline_using_nn_module.ipynb)** - Advanced training with custom neural network modules

### 🎯 Fashion-MNIST Project Series
Progressive implementations of neural networks on Fashion-MNIST dataset:

1. **[ann_fashion_mnist_pytorch.ipynb](ann_fashion_mnist_pytorch.ipynb)** - Basic artificial neural network
2. **[ann_fashion_mnist_pytorch_gpu.ipynb](ann_fashion_mnist_pytorch_gpu.ipynb)** - GPU-accelerated version
3. **[ann_fashion_mnist_pytorch_gpu_optimized.ipynb](ann_fashion_mnist_pytorch_gpu_optimized.ipynb)** - Performance optimizations
4. **[ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb](ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb)** - Automated hyperparameter tuning with Optuna

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision
pip install pandas numpy matplotlib scikit-learn
pip install optuna  # For hyperparameter optimization
pip install kagglehub  # For Fashion-MNIST dataset
pip install torchinfo  # For model summaries
```

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Raghav0079/Neural-Networks-PyTorch-.git
   cd Neural-Networks-PyTorch-
   ```

2. **Start with fundamentals**:
   Open [tensors_in_pytorch.ipynb](tensors_in_pytorch.ipynb) to learn PyTorch basics

3. **Progress through tutorials**:
   Follow the notebooks in order of complexity, from basic tensors to complete training pipelines

4. **Explore Fashion-MNIST series**:
   Work through the Fashion-MNIST implementations to see real-world applications

## 🏗️ Project Structure

```
Neural-Networks-PyTorch-/
├── 📊 Core Concepts
│   ├── tensors_in_pytorch.ipynb
│   ├── pytorch_autograd.ipynb
│   └── pytorch_nn_module.ipynb
├── 🔄 Data & Training
│   ├── dataset_and_dataloader_demo.ipynb
│   ├── pytorch_training_pipeline.ipynb
│   ├── pytorch_training_pipeline_using_dataset_and_dataloader.ipynb
│   └── pytorch_training_pipeline_using_nn_module.ipynb
├── 🎯 Fashion-MNIST Applications
│   ├── ann_fashion_mnist_pytorch.ipynb
│   ├── ann_fashion_mnist_pytorch_gpu.ipynb
│   ├── ann_fashion_mnist_pytorch_gpu_optimized.ipynb
│   └── ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb
└── 📄 Python Scripts (Generated from notebooks)
```

## 🎓 Learning Path

### Beginner Level
1. Start with [tensors_in_pytorch.ipynb](tensors_in_pytorch.ipynb) for PyTorch fundamentals
2. Learn automatic differentiation with [pytorch_autograd.ipynb](pytorch_autograd.ipynb)
3. Build your first neural network using [pytorch_nn_module.ipynb](pytorch_nn_module.ipynb)

### Intermediate Level
4. Master data handling with [dataset_and_dataloader_demo.ipynb](dataset_and_dataloader_demo.ipynb)
5. Implement complete training pipelines
6. Apply knowledge to [ann_fashion_mnist_pytorch.ipynb](ann_fashion_mnist_pytorch.ipynb)

### Advanced Level
7. Optimize for GPU performance
8. Implement automated hyperparameter tuning with Optuna
9. Build production-ready training workflows

## 💡 Key Features Covered

- **Tensor Operations**: Creation, manipulation, and GPU acceleration
- **Automatic Differentiation**: Backpropagation and gradient computation
- **Neural Network Architectures**: Custom `nn.Module` implementations
- **Data Loading**: Efficient data pipelines with `Dataset` and `DataLoader`
- **Training Loops**: Complete training and validation workflows
- **GPU Optimization**: CUDA integration and performance tuning
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Model Evaluation**: Metrics, visualization, and performance analysis

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration
- **Optuna**: Hyperparameter optimization
- **Scikit-learn**: Data preprocessing and utilities
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computations

## 📈 Performance Highlights

- GPU-accelerated training for faster convergence
- Automated hyperparameter tuning achieving optimal model performance
- Efficient data loading pipelines for large datasets
- Memory-optimized implementations for better resource utilization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Fashion-MNIST dataset creators (Zalando Research)
- Optuna team for hyperparameter optimization tools

## 📧 Contact

**Raghav** - [GitHub Profile](https://github.com/Raghav0079)

---

⭐ **Star this repository if you find it helpful!**
