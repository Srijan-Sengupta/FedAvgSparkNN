package org.cu.experiments.spark;

import org.cu.experiments.spark.nn.Sequential;
import org.cu.experiments.spark.utils.DataPoint;
import org.cu.experiments.spark.utils.MSELoss;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Client-side execution logic for FederatedAveraging (Algorithm 1 in McMahan et al.).
 */
public class Clients implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Payload returned by each participating client to the server:
     * Contains n_k * w_k and the client's local sample count n_k.
     */
    public static class ClientResult implements Serializable {
        private static final long serialVersionUID = 1L;

        public final double[] weightedWeights;
        public final long numSamples;

        public ClientResult(double[] weightedWeights, long numSamples) {
            this.weightedWeights = weightedWeights;
            this.numSamples = numSamples;
        }

        public static ClientResult merge(ClientResult a, ClientResult b) {
            double[] sum = new double[a.weightedWeights.length];
            for (int i = 0; i < sum.length; i++) {
                sum[i] = a.weightedWeights[i] + b.weightedWeights[i];
            }
            return new ClientResult(sum, a.numSamples + b.numSamples);
        }
    }

    /**
     * ClientUpdate(k, w) execution:
     * Performs E local epochs of minibatch SGD with batch size B and learning rate eta.
     */
    public static double[] clientUpdate(
            Sequential modelTemplate,
            double[] globalWeights,
            List<DataPoint> localData,
            int localEpochs,
            int batchSize,
            double learningRate,
            long seed) {

        int n = localData.size();
        if (n == 0) {
            return globalWeights.clone();
        }

        Sequential localModel = (Sequential) modelTemplate.copy();
        localModel.setFlatParameters(globalWeights);

        MSELoss lossFunc = new MSELoss();
        Random rng = new Random(seed);

        int actualBatchSize = (batchSize <= 0 || batchSize > n) ? n : batchSize;

        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }

        for (int epoch = 0; epoch < localEpochs; epoch++) {
            Collections.shuffle(indices, rng);

            for (int start = 0; start < n; start += actualBatchSize) {
                int end = Math.min(n, start + actualBatchSize);
                int currentBatchCount = end - start;

                localModel.clearGradients();

                for (int i = start; i < end; i++) {
                    DataPoint dp = localData.get(indices.get(i));
                    double[] pred = localModel.forward(dp.features);
                    double[] gradLoss = lossFunc.gradient(pred, dp.label);
                    localModel.backward(gradLoss);
                }

                localModel.updateParameters(learningRate, currentBatchCount);
            }
        }

        return localModel.getFlatParameters();
    }
}
