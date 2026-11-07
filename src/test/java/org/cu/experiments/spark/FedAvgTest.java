package org.cu.experiments.spark;

import org.cu.experiments.spark.nn.Linear;
import org.cu.experiments.spark.nn.Sequential;
import org.cu.experiments.spark.nn.Sigmoid;
import org.cu.experiments.spark.utils.DataPoint;
import org.cu.experiments.spark.utils.EvaluationMetrics;
import org.cu.experiments.spark.utils.MSELoss;
import org.cu.experiments.spark.utils.Matrix;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class FedAvgTest {

    @Test
    public void testMatrixOperations() {
        double[][] W = {
                {1.0, 2.0},
                {3.0, 4.0}
        };
        double[] x = {0.5, -1.0};
        double[] y = Matrix.multiply(W, x);

        assertEquals(2, y.length);
        assertEquals(-1.5, y[0], 1e-6);
        assertEquals(-2.5, y[1], 1e-6);

        double[] a = {1.0, 2.0};
        double[] b = {3.0, 4.0};
        double[][] outer = Matrix.outerProduct(a, b);
        assertEquals(3.0, outer[0][0], 1e-6);
        assertEquals(4.0, outer[0][1], 1e-6);
        assertEquals(6.0, outer[1][0], 1e-6);
        assertEquals(8.0, outer[1][1], 1e-6);
    }

    @Test
    public void testSequentialMLPForwardAndBackward() {
        Random rng = new Random(42L);
        int[] hiddenLayers = {4, 2};
        Sequential mlp = Sequential.createMLP(3, hiddenLayers, 1, rng);

        assertEquals(3 * 4 + 4 + 4 * 2 + 2 + 2 * 1 + 1, mlp.numParameters());

        double[] input = {0.5, -0.2, 0.8};
        double[] output = mlp.forward(input);

        assertEquals(1, output.length);
        assertTrue(output[0] >= 0.0 && output[0] <= 1.0);

        MSELoss lossFunc = new MSELoss();
        double target = 1.0;
        double lossBefore = lossFunc.loss(output, target);
        double[] gradLoss = lossFunc.gradient(output, target);

        mlp.clearGradients();
        mlp.backward(gradLoss);

        double[] grads = mlp.getFlatGradients();
        assertEquals(mlp.numParameters(), grads.length);

        // Perform SGD step
        mlp.updateParameters(0.5, 1.0);

        double[] outputAfter = mlp.forward(input);
        double lossAfter = lossFunc.loss(outputAfter, target);
        assertTrue(lossAfter < lossBefore, "One SGD step should reduce loss on the sample");
    }

    @Test
    public void testClientUpdateReducesLocalLoss() {
        Random rng = new Random(42L);
        Sequential template = Sequential.createMLP(3, new int[]{4}, 1, rng);
        double[] initialWeights = template.getFlatParameters();

        List<DataPoint> clientData = new ArrayList<>();
        clientData.add(new DataPoint(new double[]{1.0, 0.0, 1.0}, 1.0));
        clientData.add(new DataPoint(new double[]{0.0, 1.0, 0.0}, 0.0));
        clientData.add(new DataPoint(new double[]{1.0, 1.0, 0.0}, 1.0));

        double[] updatedWeights = Clients.clientUpdate(
                template,
                initialWeights,
                clientData,
                5,    // 5 local epochs
                1,    // batch size 1
                0.2,  // learning rate
                123L
        );

        assertFalse(Arrays.equals(initialWeights, updatedWeights), "Weights should be updated after clientUpdate");

        Sequential modelInitial = (Sequential) template.copy();
        modelInitial.setFlatParameters(initialWeights);

        Sequential modelUpdated = (Sequential) template.copy();
        modelUpdated.setFlatParameters(updatedWeights);

        MSELoss loss = new MSELoss();
        double initialTotalLoss = 0.0;
        double updatedTotalLoss = 0.0;

        for (DataPoint dp : clientData) {
            initialTotalLoss += loss.loss(modelInitial.forward(dp.features), dp.label);
            updatedTotalLoss += loss.loss(modelUpdated.forward(dp.features), dp.label);
        }

        assertTrue(updatedTotalLoss < initialTotalLoss, "ClientUpdate should decrease total local loss");
    }

    @Test
    public void testClientResultMerge() {
        double[] w1 = {2.0, 4.0};
        double[] w2 = {6.0, 8.0};

        Clients.ClientResult cr1 = new Clients.ClientResult(w1, 10);
        Clients.ClientResult cr2 = new Clients.ClientResult(w2, 20);

        Clients.ClientResult merged = Clients.ClientResult.merge(cr1, cr2);

        assertEquals(30, merged.numSamples);
        assertEquals(8.0, merged.weightedWeights[0], 1e-6);
        assertEquals(12.0, merged.weightedWeights[1], 1e-6);

        // Average
        double[] avg = new double[2];
        for (int i = 0; i < 2; i++) {
            avg[i] = merged.weightedWeights[i] / merged.numSamples;
        }
        // (2*10 + 6*20)/30 = 140/30 = 4.6666... Wait! Here w1 and w2 were already weighted (n*w):
        // (2 + 6)/30 = 8/30 = 0.2666...
        assertEquals(8.0 / 30.0, avg[0], 1e-6);
        assertEquals(12.0 / 30.0, avg[1], 1e-6);
    }

    @Test
    public void testEvaluationMetrics() {
        EvaluationMetrics m1 = new EvaluationMetrics(10, 2, 8, 0, 1.5, 20);
        EvaluationMetrics m2 = new EvaluationMetrics(5, 1, 12, 2, 0.5, 20);

        EvaluationMetrics combined = EvaluationMetrics.merge(m1, m2);
        assertEquals(15, combined.tp);
        assertEquals(3, combined.fp);
        assertEquals(20, combined.tn);
        assertEquals(2, combined.fn);
        assertEquals(40, combined.count);
        assertEquals(2.0, combined.totalLoss, 1e-6);
        assertEquals(0.05, combined.averageLoss(), 1e-6);
        assertEquals((15.0 + 20.0) / 40.0, combined.accuracy(), 1e-6);
    }
}
