package org.cu.experiments.spark.utils;

import java.io.Serializable;

/**
 * Mean Squared Error loss: L = 1/2 * (y_pred - y_true)^2
 */
public class MSELoss implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Computes the scalar loss for a single output neuron and target.
     */
    public double loss(double[] prediction, double target) {
        double diff = prediction[0] - target;
        return 0.5 * diff * diff;
    }

    /**
     * Computes the derivative dL/dy_pred = (y_pred - y_true).
     */
    public double[] gradient(double[] prediction, double target) {
        return new double[] { prediction[0] - target };
    }
}
