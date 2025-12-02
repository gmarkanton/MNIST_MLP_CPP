# MNIST MLP — Neural Network from Scratch in Modern C++

This project implements a fully manual Multilayer Perceptron for MNIST classification using only C++17 STL and OpenCV for visualization.  
All components — forward pass, backward pass, weight updates, batching, and initialization — are written from scratch without any machine-learning frameworks.

---

# Features
- Custom MLP architecture: 784 → 256 → 64 → 10
- He initialization
- ReLU / LeakyReLU activation
- Manual forward pass & backpropagation
- Mini-batch gradient descent
- Gradient clipping
- Save/load weights
- MNIST image visualization with OpenCV

---

# Project Structure
src/
mlp.cpp → MLP implementation
layer.hpp → Dense layer
utils.hpp → MNIST loader & helpers
main.cpp → Entry point

data/
(empty by default — MNIST not included)

CMakeLists.txt
README.md
LICENSE


---

# Build Instructions

# Requirements
- C++17
- CMake ≥ 3.10
- OpenCV

# Build
mkdir build
cd build
cmake 


---

# Run Training
./mnist_mlp --train



# Run Testing
./mnist_mlp --test

---

# MNIST Dataset

Download the four official MNIST files from:

https://yann.lecun.com/exdb/mnist/

Place them in the `data/` directory:

data/train-images.idx3-ubyte
data/train-labels.idx1-ubyte
data/t10k-images.idx3-ubyte
data/t10k-labels.idx1-ubyte


*(The dataset is not included in this repository due to licensing restrictions.)*

---

## 📄 License
MIT License
