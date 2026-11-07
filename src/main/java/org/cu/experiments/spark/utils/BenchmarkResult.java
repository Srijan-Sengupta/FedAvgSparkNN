package org.cu.experiments.spark.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Stores benchmark metrics for performance comparison.
 */
public class BenchmarkResult implements Serializable {
    private static final long serialVersionUID = 1L;

    public static class StepRecord implements Serializable {
        private static final long serialVersionUID = 1L;

        public final int step;
        public final double timeSeconds;
        public final double testAccuracy;
        public final double testLoss;
        public final double commMB;

        public StepRecord(int step, double timeSeconds, double testAccuracy, double testLoss, double commMB) {
            this.step = step;
            this.timeSeconds = timeSeconds;
            this.testAccuracy = testAccuracy;
            this.testLoss = testLoss;
            this.commMB = commMB;
        }
    }

    public final String algorithmName;
    public double totalTimeSeconds;
    public double totalCommMB;
    public int totalRoundsOrEpochs;
    public EvaluationMetrics finalMetrics;
    public final List<StepRecord> stepHistory;

    public BenchmarkResult(String algorithmName) {
        this.algorithmName = algorithmName;
        this.stepHistory = new ArrayList<>();
    }

    public void addStep(int step, double timeSec, double testAcc, double testLoss, double commMB) {
        this.stepHistory.add(new StepRecord(step, timeSec, testAcc, testLoss, commMB));
    }
}
