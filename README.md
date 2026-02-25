# Quad_vs_Grad
First-Order vs Second-Order Optimization Methods for Deep Learning

This repository demonstrates and visualizes the differences between first-order (e.g., SGD, Adam) and second-order (e.g., Newton's method, BFGS, L-BFGS) optimization techniques in a deep learning context. It uses PyTorch, NumPy, SciPy, and Matplotlib to compare their optimization paths on a non-convex loss landscape.

The content and insights here are inspired by the University of Tübingen's "Numerics of Machine Learning" lecture slides (2022/23), available at the [Numerics of ML repo](https://github.com/philipphennig/NumericsOfML.git), specifically the second-order optimization slide: [12_SecondOrderOptimization.pdf](https://github.com/philipphennig/NumericsOfML/blob/main/slides/12_SecondOrderOptimization.pdf). **Grok 4.1 Fast** was used for grammar correction of this `README.md`.

## Advantages of Second-Order Methods
Second-order optimization methods, like Newton's method, offer several appealing benefits, particularly when compared to first-order (gradient-based) approaches:
- They are robust to ill-conditioned problems, handling steep or flat landscapes effectively.
- They converge quickly near the optimum, often achieving quadratic convergence rates.
- Newton steps inherently have a "natural length", avoiding the need for manual step-size tuning common in gradient descent.
- They have a proven track record and are the default in many scientific domains outside deep learning.

Despite these strengths, their application in deep learning is **limited** due to three core challenges: *non-convexity*, *computational cost*, and *stochasticity*. Below is a summary of each problem, why it's an issue in deep learning contexts, and the proposed solutions or workarounds from the University of Tübingen's "Numerics of ML" slides.

### 1. Non-Convexity
**Problem**: Deep learning objectives are typically non-convex, meaning the quadratic approximation (used in Newton's method) can be poor or misleading far from the optimum. Even in convex cases, the model might fail, leading to unreliable updates or divergence.

**Why It's a Challenge in DL**: Neural networks have high-dimensional, non-convex loss landscapes with saddle points and local minima. Standard Newton steps assume positive definite curvature, but negative curvature can point toward worse solutions.

**Answers/Solutions**:
- **Positive Semi-Definite Curvature Matrices (e.g., GGN and FIM)**: Use proxies like the Generalized Gauss-Newton (GGN) matrix or Fisher Information Matrix (FIM) for "meaningful" curvature approximations. These are always positive semi-definite, and often GGN equals FIM. They provide unbiased estimates even on finite datasets.
- **Damping**: Add regularization to control update conservatism. Heuristics like Levenberg-Marquardt damping work well in practice by blending Newton-like steps with safer gradient descent behaviors.

### 2. Cost (Accessing and Inverting the Curvature Matrix)
**Problem**: Computing a Newton step requires the Hessian (curvature) matrix, which is massive in deep learning (storage scales as **O(D^2)**, inversion as **O(D^3)**, where **D** is the number of parameters, often millions or billions).

**Why It's a Challenge in DL**: Direct computation is infeasible due to memory and time constraints. We need efficient ways to approximate, access, and "invert" this matrix without full materialization.

**Answers/Solutions**:
- **BFGS & L-BFGS**: Quasi-Newton methods that build a dynamic low-rank approximation of the Hessian using gradient history. L-BFGS is memory-efficient and the default for small deterministic problems (e.g., in **SciPy**'s `optimize.minimize`).
- **HF (Hessian-Free Optimization)**: Approximates Newton steps without storing the full Hessian, using iterative methods like conjugate gradients. It's close to true Newton behavior, requires little memory, but involves more sequential computations and can be problematic with layers like BatchNorm.
- **K-FAC (Kronecker-Factored Approximate Curvature)**: A lightweight approximation of the FIM using Kronecker products for block-diagonal structure. Widely used in uncertainty quantification (e.g., Bayesian DL), though it relies heavily on heuristic damping for stability.

These approaches are pragmatic, with pros (e.g., efficiency) and cons (e.g., approximations may not always capture full curvature).

### 3. Stochasticity
**Problem**: Deep learning uses stochastic gradients (e.g., from mini-batches) rather than exact ones, so we only have noisy estimates Ĥ and ĝ instead of the true Hessian H and gradient g. This violates the assumptions of deterministic second-order methods.

**Why It's a Challenge in DL**: Finite data introduces variance, making updates unreliable. Exact computations would require full-dataset passes, which is impractical for large-scale training.

**Answers/Solutions**:
- Fundamentally unsolved.

## Project Overview
The deep learning task here is **binary classification on a 2D XOR-like grid** using a small multi-layer perceptron (MLP).  

We generate a synthetic dataset of points distributed on a 2D grid in [-1, 1] × [-1, 1], where each point is labeled according to the classic XOR pattern (e.g., positive label if **(sign(x) XOR sign(y))** is true, or quadrant-based XOR logic). This creates a highly non-linear, non-convex classification problem - the famous task that single-layer perceptrons cannot solve.

A tiny MLP (e.g., 2 input -> small hidden layer -> 1 output, with only a few trainable parameters) is used so that the parameter space remains low-dimensional (typically visualized in 2D by fixing most weights and varying two). The model is trained to minimize binary cross-entropy loss, and we record the paths taken by different optimizers through the loss landscape.

The project compares:
- **First-order methods**: SGD and Adam
- **Second-order / quasi-second-order methods**: Newton's method, BFGS, L-BFGS, and approximations like GGN/FIM/K-FAC (adapted for this small setting)

Optimization paths are visualized on contour plots of the loss surface - both as static images and animated GIFs - clearly showing how second-order methods exploit curvature for faster or more direct convergence, while first-order methods follow gradient directions more cautiously.

- **Dependencies**: Matplotlib (plots and animations), NumPy (arrays), SciPy (optimization helpers), Pillow (for saving GIFs), PyTorch (for autograd, models, and optimizers).
- **Outputs**: `paths.png` (static overlay of paths) and `journey.gif` (animated paths).

## Project Structure
```
Quad_vs_Grad/
├── main.py             # Entry point: Runs optimizations and generates visuals
├── optimizers.py       # Implements first- and second-order optimizers
├── visualize.py        # Functions for static plots and animations
├── data.py             # Dataset generation for XOR-like grid
├── requirements.txt    # Dependencies: matplotlib, numpy, pillow, scipy, torch
└── README.md           
```

After running, you'll also see generated files like `paths.png` and `journey.gif` in the root.

## How to Run

Follow these steps to get up and running quickly:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Stochastic-Batman/Quad_vs_Grad.git
   ```

2. **Navigate into the project folder**  
   ```bash
   cd Quad_vs_Grad
   ```

3. **Create and activate a virtual environment** (recommended)  
   ```bash
   python -m venv qvg_venv
   ```
   - On Linux/macOS:  
     ```bash
     source qvg_venv/bin/activate
     ```
   - On Windows:  
     ```bash
     .\qvg_venv\Scripts\activate
     ```

4. **Install the required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the main script**  
   The simplest way:  
   ```bash
   python main.py
   ```

   Customize the run with command-line arguments:  
   ```bash
   python main.py --x0 0.5 --y0 -0.8 --lr 0.05 --steps 500 --tol 1e-5
   ```

   Available flags include:
   - `--x0` / `--y0` : Starting point in parameter space (default: [2.0, -2.0])
   - `--lr` : Learning rate for first-order methods (default depends on optimizer)
   - `--steps` : Maximum number of optimization steps (default: 500-1000)
   - `--tol` : Convergence tolerance (default: 1e-6)
   - `--damping` : Damping factor for Newton-like methods

6. **View the results**  
   After the script finishes, open these files in the project root:
   - `paths.png`: static contour plot showing the full optimization paths overlaid on the loss surface
   - `journey.gif`: animated version of the paths, showing step-by-step movement

Have fun experimenting with different starting points, learning rates, and optimizer combinations - the differences between first- and second-order methods become very clear on this classic non-linear problem!
