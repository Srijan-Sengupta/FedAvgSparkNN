package org.cu.experiments.spark.nn;

import java.io.Serializable;

/**
 * Base interface for all neural network modules.
 */
public interface Module extends Serializable {

    /**
     * Performs the forward pass.
     * @param input input vector
     * @return activation vector
     */
    double[] forward(double[] input);

    /**
     * Performs the backward pass.
     * @param gradOutput gradient of the loss with respect to this module's output
     * @return gradient of the loss with respect to this module's input
     */
    double[] backward(double[] gradOutput);

    /**
     * Total number of learnable parameters in this module.
     */
    int numParameters();

    /**
     * Extracts learnable parameters into a flat array starting at offset.
     * @param target array to populate
     * @param offset starting index in target
     * @return next offset after copying this module's parameters
     */
    int getParameters(double[] target, int offset);

    /**
     * Sets learnable parameters from a flat array starting at offset.
     * @param source array of parameter values
     * @param offset starting index in source
     * @return next offset after consuming this module's parameters
     */
    int setParameters(double[] source, int offset);

    /**
     * Extracts accumulated gradients into a flat array starting at offset.
     * @param target array to populate
     * @param offset starting index in target
     * @return next offset after copying this module's gradients
     */
    int getGradients(double[] target, int offset);

    /**
     * Updates internal parameters using accumulated gradients with SGD:
     * theta = theta - (lr / batchSize) * accumulatedGrad
     */
    void updateParameters(double learningRate, double batchSize);

    /**
     * Clears/resets accumulated gradients to zero.
     */
    void clearGradients();

    /**
     * Creates a deep clone of this module (including weights).
     */
    Module copy();
}
