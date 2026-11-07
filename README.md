# FederatedAveraging (FedAvg) vs. Spark Distributed NN Benchmark

This project implements and benchmarks the **FederatedAveraging (`FedAvg`)** algorithm from the seminal paper:
> **"Communication-Efficient Learning of Deep Networks from Decentralized Data"**  
> *H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas (AISTATS 2017 / arXiv:1602.05629)*

against the **Spark Distributed NN** approach (data-parallel synchronous gradient descent via MapReduce from `HiggsMapReduceNN`).

---

## Architecture Overview

```
                                      ┌─────────────────────────────────────┐
                                      │             Coordinating            │
                                      │            Server (Driver)          │
                                      └──────────────────┬──────────────────┘
                                                         │
                             Broadcast Global Model w_t  │  Collect w_{t+1}^k (FedAvg)
                                                         │  or Gradients (MapReduce)
                                                         ▼
       ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼                         ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│   Client 0   │          │   Client 1   │          │   Client 2   │          │  Client K-1  │
│ (Partition 0)│          │ (Partition 1)│          │ (Partition 2)│          │(Partition K-1│
│              │          │              │          │              │          │              │
│ Local Data P0│          │ Local Data P1│          │ Local Data P2│          │ Local Data PK│
│ E epochs,    │          │ E epochs,    │          │ E epochs,    │          │ E epochs,    │
│ Batch size B │          │ Batch size B │          │ Batch size B │          │ Batch size B │
└──────────────┘          └──────────────┘          └──────────────┘          └──────────────┘
```

### Module Structure (`org.cu.experiments.spark`)

* **`nn`**:
  * [`Module.java`](src/main/java/org/cu/experiments/spark/nn/Module.java): Base interface for neural network layers (`forward`, `backward`, parameter serialization, gradient management).
  * [`Linear.java`](src/main/java/org/cu/experiments/spark/nn/Linear.java): Dense fully-connected layer ($y = Wx + b$) initialized with Gaussian weights scaled by $1 / \sqrt{\text{fan\_in}}$.
  * [`Sigmoid.java`](src/main/java/org/cu/experiments/spark/nn/Sigmoid.java): Element-wise sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  * [`Sequential.java`](src/main/java/org/cu/experiments/spark/nn/Sequential.java): Sequential container chaining linear and activation modules to construct the MLP.
* **`utils`**:
  * [`Matrix.java`](src/main/java/org/cu/experiments/spark/utils/Matrix.java): Linear algebra operations (matrix-vector multiplication, outer products, transpose multiplication).
  * [`MSELoss.java`](src/main/java/org/cu/experiments/spark/utils/MSELoss.java): Mean Squared Error loss and derivative computation.
  * [`DataPoint.java`](src/main/java/org/cu/experiments/spark/utils/DataPoint.java): Serializable dataset sample container `(features[], label)`.
  * [`EvaluationMetrics.java`](src/main/java/org/cu/experiments/spark/utils/EvaluationMetrics.java): Calculates and prints Confusion Matrix, Accuracy, Precision, Recall, Specificity, and F1 Score.
  * [`BenchmarkResult.java`](src/main/java/org/cu/experiments/spark/utils/BenchmarkResult.java): Records step-by-step history of loss, accuracy, wall-clock time, and communication overhead.
* **Core Execution**:
  * [`Clients.java`](src/main/java/org/cu/experiments/spark/Clients.java): Implements `ClientUpdate(k, w)` (Algorithm 1 from paper), executing $E$ local epochs over mini-batches of size $B$.
  * [`Server.java`](src/main/java/org/cu/experiments/spark/Server.java): Implements the coordinating server for both `FedAvg` (client sampling, model broadcast, weighted model averaging) and `Spark Distributed NN` (synchronous MapReduce mini-batch gradient aggregation).
  * [`Main.java`](src/main/java/org/cu/experiments/spark/Main.java): Main entry point: data loading, train/test split, feature standardization, benchmark execution, and head-to-head comparison report.

---

## Running the Benchmark

### 1. Build and Run Unit Tests
```bash
./gradlew test
```

### 2. Run the Head-to-Head Comparison
```bash
# Quick benchmark on 50,000 samples
./gradlew run --args="--max-rows=50000 --rounds=10 --epochs=5 --clients=50"

# Scaled benchmark on 200,000 samples
./gradlew run --args="--max-rows=200000 --rounds=20 --epochs=10 --clients=100"

# Run only FedAvg
./gradlew run --args="--mode=fedavg --max-rows=100000 --rounds=20"

# Run only MapReduce NN
./gradlew run --args="--mode=mapreduce --max-rows=100000 --epochs=10"
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--path=<file>` | `HIGGS.csv` | Path to HIGGS dataset CSV |
| `--max-rows=<N>` | `200000` | Number of rows to read (`0` or negative for all 11M rows) |
| `--mode=<both\|fedavg\|mapreduce>` | `both` | Which algorithm to execute |
| `--rounds=<N>` | `20` | Total communication rounds for FedAvg |
| `--clients=<K>` | `100` | Total number of client partitions $K$ |
| `--epochs=<N>` | `10` | Total training epochs for Spark MapReduce NN |
