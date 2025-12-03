# C++ Multi-Layer Perceptron (MLP) for MNIST Classification

## Project Summary

This repository presents a **Multi-Layer Perceptron (MLP) implementation built entirely from scratch in C++**. The project's primary goal is to classify handwritten digits from the standard **MNIST dataset**. By implementing all fundamental mechanisms—from data loading to gradient updates—without relying on high-level deep learning frameworks (e.g., PyTorch, TensorFlow), this code demonstrates a deep, low-level understanding of both C++ efficiency and the mathematical core of neural networks.

---

## Technical Highlights

The implementation showcases robust and advanced features critical for modern machine learning:

* **Custom Core Engine:** Full, manual implementation of the **Forward Pass** and the **Backpropagation** algorithm, calculating all gradients (dL/dW, dL/dBias) directly in C++.
* **Robust Training Mechanics:**
    * **Weight Initialization:** Uses **He Initialization** suitable for ReLU-based networks.
    * **Activation:** Implements the **Leaky ReLU** function.
    * **Optimization:** Supports **Mini-Batch Gradient Descent**.
    * **Stability:** Includes **Softmax** with numerical stability (using the log-sum-exp trick) and **Cross-Entropy Loss**.
    * **Mitigation:** Features **Gradient Clipping** (L2-norm) to manage exploding gradients.
* **Low-Level Data Handling:** Implements a native reader for the MNIST **binary file format**, correctly handling **Endianness** (`__builtin_bswap32`) for data integrity.
* **Visualization:** Integration with **OpenCV** allows for displaying the input image alongside the network's prediction and loss statistics.

---

## Network Architecture

The current configuration uses a network structure commonly suitable for the MNIST task:

| Layer Type | Input Size | Output Size | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | 784 (28x28 pixels) | 784 | N/A |
| **Hidden Layer 1** | 784 | 256 | Leaky ReLU |
| **Hidden Layer 2** | 256 | 64 | Leaky ReLU |
| **Output Layer** | 64 | 10 | Softmax |

**Hyperparameters:**
* **Learning Rate ($\eta$):** $0.0001$
* **Batch Size:** 32
* **Loss Function:** Cross-Entropy Loss

---

## Getting Started

### Prerequisites

1.  **C++ Environment:** A C++ compiler supporting C++11 or later (Tested with GCC/Clang via CLion/CMake).
2.  **OpenCV Library:** A local installation of the **OpenCV** library is required and must be correctly linked via the project's `CMakeLists.txt` file.
3.  **MNIST Dataset:** The original four binary files must be downloaded and placed into the **`data/MNIST`** directory inside the project root:
    * `train-images.idx3-ubyte`
    * `train-labels.idx1-ubyte`
    * `t10k-images.idx3-ubyte`
    * `t10k-labels.idx1-ubyte`

### Build and Execution

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    ```
2.  **Build:** Open the project in CLion. CMake should automatically configure the build process based on `CMakeLists.txt`.
3.  **Run:** Execute the compiled program. The application is command-line driven, prompting the user to select the operating mode (**TRAIN** or **TEST**) and configuration details (e.g., weight initialization or loading).

---
[Future Work] The current monolithic structure will be refactored into modular C++ header/source files for improved maintainability.
## License

This project is licensed under the MIT License.
