# Sparse Recovery using Modified Genetic Algorithm

## 1. Problem Formulation

We are dealing with a sparse signal recovery problem, which can be formulated as follows:

y = Hx + n

Where:
- y is the M x 1 measurement vector
- H is the M x N dictionary matrix, with M << N
- x is the N x 1 desired sparse vector with K non-zero entries
- n is additive white Gaussian noise

Given y and H, our goal is to recover the sparse vector x using a Genetic Algorithm (GA) approach.

### Key Challenges:
1. Underdetermined system (M << N)
2. Sparsity constraint on x
3. Presence of noise in measurements

## 2. Proposed Solution: Modified Genetic Algorithm

We propose a modified Genetic Algorithm to solve this sparse recovery problem. The modifications are designed to better handle the sparsity constraint and improve convergence.

### Key Modifications:
1. **Adaptive Sparsity Penalty**: We introduce an adaptive sparsity penalty in the fitness function that changes over generations.
2. **Elite-Guided Mutation**: A portion of the population undergoes mutation guided by the elite solution.
3. **Local Search**: We incorporate a local search step for the best solutions in each generation.


## 3. Execution and Results

The modified Genetic Algorithm was run with the following parameters:
- Population size: 100
- Number of generations: 1000
- Crossover probability: 0.7
- Mutation probability: 0.2

### Key Findings:

1. **Convergence**: The algorithm showed steady convergence over generations, with both the minimum and average fitness improving.

2. **Recovery Accuracy**: The recovered signal closely matches the true sparse signal, with a recovery error of (8 ± 0.5).

3. **Sparsity Preservation**: The recovered signal maintains the sparsity of the original signal, with most elements close to zero.

4. **Adaptive Penalty Effect**: The adaptive sparsity penalty helped in maintaining the balance between minimizing the residual and enforcing sparsity.

5. **Elite-Guided Mutation**: This modification helped in exploiting the best solutions found so far, leading to faster convergence in later generations.

6. **Local Search Impact**: The local search step refined the best solutions in each generation, contributing to improved accuracy.

## 4. Visualizations


1. **True vs Recovered Signal Plot**: 
    ![image]("https://github.com/Abhishekdx300/Sparse-recovery-py/sparse_recovery_output1/true_recovered_20241010_190257.png")

2. **Fitness Over Generations Plot**:
    ![image]("https://github.com/Abhishekdx300/Sparse-recovery-py/sparse_recovery_output1/fitness_20241010_190257.png")


3. **Recovery Error Plot**:
    ![image]("https://github.com/Abhishekdx300/Sparse-recovery-py/sparse_recovery_output1/error_20241010_190257.png")



## 5. Conclusions

1. **Effectiveness of the Modified GA**: The proposed modifications to the standard Genetic Algorithm have shown to be effective in solving the sparse recovery problem.

2. **Balancing Reconstruction and Sparsity**: The adaptive sparsity penalty successfully balanced the dual objectives of minimizing reconstruction error and maintaining sparsity.

3. **Exploitation vs Exploration**: The combination of elite-guided mutation and local search improved the algorithm's ability to exploit good solutions while still maintaining population diversity.

4. **Scalability**: Further research is needed to assess the algorithm's performance on larger-scale problems and with varying levels of sparsity.

5. **Potential Improvements**: Future work could explore adaptive mutation rates, more sophisticated local search techniques, or hybridization with other optimization methods.

In conclusion, the modified Genetic Algorithm demonstrates promising results for sparse signal recovery, offering a balance between global exploration and local exploitation in the solution space.