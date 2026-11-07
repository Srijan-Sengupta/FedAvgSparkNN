package org.cu.experiments.spark.utils;

import java.io.Serializable;
import java.util.Random;

/**
 * Linear algebra utilities for neural network operations.
 */
public final class Matrix implements Serializable {
    private static final long serialVersionUID = 1L;

    private Matrix() {}

    /**
     * Matrix-vector multiplication: y = W * x
     * @param W weight matrix of shape [rows][cols]
     * @param x input vector of length cols
     * @return output vector of length rows
     */
    public static double[] multiply(double[][] W, double[] x) {
        int rows = W.length;
        int cols = x.length;
        double[] y = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += W[i][j] * x[j];
            }
            y[i] = sum;
        }
        return y;
    }

    /**
     * Vector outer product: M = a * b^T
     * @param a vector of length rows
     * @param b vector of length cols
     * @return matrix of shape [rows][cols]
     */
    public static double[][] outerProduct(double[] a, double[] b) {
        int rows = a.length;
        int cols = b.length;
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = a[i] * b[j];
            }
        }
        return m;
    }

    /**
     * Transposed matrix-vector multiplication: y = W^T * delta
     * @param W weight matrix of shape [rows][cols]
     * @param delta vector of length rows
     * @return vector of length cols
     */
    public static double[] multiplyTranspose(double[][] W, double[] delta) {
        int rows = W.length;
        int cols = W[0].length;
        double[] result = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += W[i][j] * delta[i];
            }
            result[j] = sum;
        }
        return result;
    }

    /**
     * Initializes a 2D weight matrix with Gaussian values scaled by 1/sqrt(fanIn).
     */
    public static double[][] initGaussian(int rows, int cols, Random rng) {
        double[][] m = new double[rows][cols];
        double scale = Math.sqrt(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = rng.nextGaussian() / scale;
            }
        }
        return m;
    }

    /**
     * Element-wise addition of two 2D matrices.
     */
    public static double[][] add(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] res = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res[i][j] = a[i][j] + b[i][j];
            }
        }
        return res;
    }

    /**
     * Element-wise addition of two vectors.
     */
    public static double[] add(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i] + b[i];
        }
        return res;
    }

    /**
     * Deep copy of a 2D matrix.
     */
    public static double[][] copy(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = m[i].clone();
        }
        return c;
    }
}
