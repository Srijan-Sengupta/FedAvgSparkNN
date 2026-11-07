package org.cu.experiments.spark.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Sequential container chaining multiple neural network modules.
 */
public class Sequential implements Module {
    private static final long serialVersionUID = 1L;

    public final List<Module> modules;

    public Sequential() {
        this.modules = new ArrayList<>();
    }

    public Sequential(Module... modules) {
        this.modules = new ArrayList<>(Arrays.asList(modules));
    }

    public Sequential(List<Module> modules) {
        this.modules = new ArrayList<>(modules);
    }

    public void add(Module module) {
        this.modules.add(module);
    }

    /**
     * Factory to build an MLP matching the Higgs benchmark architecture:
     * Linear -> Sigmoid -> ... -> Linear -> Sigmoid
     */
    public static Sequential createMLP(int inFeatures, int[] hiddenLayers, int outFeatures, Random rng) {
        Sequential seq = new Sequential();
        int currentIn = inFeatures;
        for (int hidden : hiddenLayers) {
            seq.add(new Linear(currentIn, hidden, rng));
            seq.add(new Sigmoid());
            currentIn = hidden;
        }
        seq.add(new Linear(currentIn, outFeatures, rng));
        seq.add(new Sigmoid());
        return seq;
    }

    @Override
    public double[] forward(double[] input) {
        double[] current = input;
        for (Module m : modules) {
            current = m.forward(current);
        }
        return current;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        double[] currentGrad = gradOutput;
        for (int i = modules.size() - 1; i >= 0; i--) {
            currentGrad = modules.get(i).backward(currentGrad);
        }
        return currentGrad;
    }

    @Override
    public int numParameters() {
        int total = 0;
        for (Module m : modules) {
            total += m.numParameters();
        }
        return total;
    }

    public double[] getFlatParameters() {
        double[] params = new double[numParameters()];
        getParameters(params, 0);
        return params;
    }

    public void setFlatParameters(double[] params) {
        setParameters(params, 0);
    }

    public double[] getFlatGradients() {
        double[] grads = new double[numParameters()];
        getGradients(grads, 0);
        return grads;
    }

    @Override
    public int getParameters(double[] target, int offset) {
        for (Module m : modules) {
            offset = m.getParameters(target, offset);
        }
        return offset;
    }

    @Override
    public int setParameters(double[] source, int offset) {
        for (Module m : modules) {
            offset = m.setParameters(source, offset);
        }
        return offset;
    }

    @Override
    public int getGradients(double[] target, int offset) {
        for (Module m : modules) {
            offset = m.getGradients(target, offset);
        }
        return offset;
    }

    @Override
    public void updateParameters(double learningRate, double batchSize) {
        for (Module m : modules) {
            m.updateParameters(learningRate, batchSize);
        }
    }

    @Override
    public void clearGradients() {
        for (Module m : modules) {
            m.clearGradients();
        }
    }

    @Override
    public Module copy() {
        List<Module> copied = new ArrayList<>(modules.size());
        for (Module m : modules) {
            copied.add(m.copy());
        }
        return new Sequential(copied);
    }
}
