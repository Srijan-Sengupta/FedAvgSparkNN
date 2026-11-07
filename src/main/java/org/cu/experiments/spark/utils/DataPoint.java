package org.cu.experiments.spark.utils;

import java.io.Serializable;

/**
 * Representation of a single dataset sample with features and a target label.
 */
public class DataPoint implements Serializable {
    private static final long serialVersionUID = 1L;

    public final double[] features;
    public final double label;

    public DataPoint(double[] features, double label) {
        this.features = features;
        this.label = label;
    }
}
