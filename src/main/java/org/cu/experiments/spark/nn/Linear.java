package org.cu.experiments.spark.nn;

import org.cu.experiments.spark.utils.Matrix;

import java.util.Arrays;
import java.util.Random;

/**
 * Fully connected (dense) linear layer: y = W * x + b
 */
public class Linear implements Module {
    private static final long serialVersionUID = 1L;

    public final int inFeatures;
    public final int outFeatures;

    public double[][] weights;    // [outFeatures][inFeatures]
    public double[] biases;       // [outFeatures]

    public double[][] gradWeights;
    public double[] gradBiases;

    private transient double[] lastInput;

    public Linear(int inFeatures, int outFeatures, Random rng) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weights = Matrix.initGaussian(outFeatures, inFeatures, rng);
        this.biases = new double[outFeatures];
        this.gradWeights = new double[outFeatures][inFeatures];
        this.gradBiases = new double[outFeatures];
    }

    public Linear(int inFeatures, int outFeatures, double[][] weights, double[] biases) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weights = Matrix.copy(weights);
        this.biases = biases.clone();
        this.gradWeights = new double[outFeatures][inFeatures];
        this.gradBiases = new double[outFeatures];
    }

    @Override
    public double[] forward(double[] input) {
        this.lastInput = input.clone();
        double[] output = new double[outFeatures];
        for (int i = 0; i < outFeatures; i++) {
            double sum = biases[i];
            for (int j = 0; j < inFeatures; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        if (gradWeights == null) {
            gradWeights = new double[outFeatures][inFeatures];
            gradBiases = new double[outFeatures];
        }

        // Accumulate gradients for weights and biases
        for (int i = 0; i < outFeatures; i++) {
            double g = gradOutput[i];
            gradBiases[i] += g;
            for (int j = 0; j < inFeatures; j++) {
                gradWeights[i][j] += g * lastInput[j];
            }
        }

        // Gradient with respect to layer input: dL/dx = W^T * gradOutput
        double[] gradInput = new double[inFeatures];
        for (int j = 0; j < inFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < outFeatures; i++) {
                sum += weights[i][j] * gradOutput[i];
            }
            gradInput[j] = sum;
        }
        return gradInput;
    }

    @Override
    public int numParameters() {
        return outFeatures * inFeatures + outFeatures;
    }

    @Override
    public int getParameters(double[] target, int offset) {
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                target[offset++] = weights[i][j];
            }
        }
        for (int i = 0; i < outFeatures; i++) {
            target[offset++] = biases[i];
        }
        return offset;
    }

    @Override
    public int setParameters(double[] source, int offset) {
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                weights[i][j] = source[offset++];
            }
        }
        for (int i = 0; i < outFeatures; i++) {
            biases[i] = source[offset++];
        }
        return offset;
    }

    @Override
    public int getGradients(double[] target, int offset) {
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                target[offset++] = gradWeights[i][j];
            }
        }
        for (int i = 0; i < outFeatures; i++) {
            target[offset++] = gradBiases[i];
        }
        return offset;
    }

    @Override
    public void updateParameters(double learningRate, double batchSize) {
        double scale = learningRate / batchSize;
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                weights[i][j] -= scale * gradWeights[i][j];
            }
            biases[i] -= scale * gradBiases[i];
        }
    }

    @Override
    public void clearGradients() {
        if (gradWeights != null) {
            for (int i = 0; i < outFeatures; i++) {
                Arrays.fill(gradWeights[i], 0.0);
            }
        }
        if (gradBiases != null) {
            Arrays.fill(gradBiases, 0.0);
        }
    }

    @Override
    public Module copy() {
        return new Linear(inFeatures, outFeatures, weights, biases);
    }
}
