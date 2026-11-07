package org.cu.experiments.spark.nn;

/**
 * Element-wise Sigmoid activation function: sigma(z) = 1 / (1 + exp(-z))
 */
public class Sigmoid implements Module {
    private static final long serialVersionUID = 1L;

    private transient double[] lastOutput;

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double[] forward(double[] input) {
        lastOutput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            lastOutput[i] = sigmoid(input[i]);
        }
        return lastOutput;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            double a = lastOutput[i];
            gradInput[i] = gradOutput[i] * a * (1.0 - a);
        }
        return gradInput;
    }

    @Override
    public int numParameters() {
        return 0;
    }

    @Override
    public int getParameters(double[] target, int offset) {
        return offset;
    }

    @Override
    public int setParameters(double[] source, int offset) {
        return offset;
    }

    @Override
    public int getGradients(double[] target, int offset) {
        return offset;
    }

    @Override
    public void updateParameters(double learningRate, double batchSize) {
        // No parameters
    }

    @Override
    public void clearGradients() {
        // No parameters
    }

    @Override
    public Module copy() {
        return new Sigmoid();
    }
}
