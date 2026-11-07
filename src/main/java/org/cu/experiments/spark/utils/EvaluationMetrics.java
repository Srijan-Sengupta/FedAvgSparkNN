package org.cu.experiments.spark.utils;

import java.io.Serializable;

/**
 * Encapsulates classification performance metrics.
 */
public class EvaluationMetrics implements Serializable {
    private static final long serialVersionUID = 1L;

    public final long tp;
    public final long fp;
    public final long tn;
    public final long fn;
    public final double totalLoss;
    public final long count;

    public EvaluationMetrics(long tp, long fp, long tn, long fn, double totalLoss, long count) {
        this.tp = tp;
        this.fp = fp;
        this.tn = tn;
        this.fn = fn;
        this.totalLoss = totalLoss;
        this.count = count;
    }

    public static EvaluationMetrics empty() {
        return new EvaluationMetrics(0, 0, 0, 0, 0.0, 0);
    }

    public static EvaluationMetrics merge(EvaluationMetrics a, EvaluationMetrics b) {
        return new EvaluationMetrics(
                a.tp + b.tp,
                a.fp + b.fp,
                a.tn + b.tn,
                a.fn + b.fn,
                a.totalLoss + b.totalLoss,
                a.count + b.count
        );
    }

    public double accuracy() {
        long total = tp + fp + tn + fn;
        return total == 0 ? 0.0 : (double) (tp + tn) / total;
    }

    public double precision() {
        return (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
    }

    public double recall() {
        return (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
    }

    public double specificity() {
        return (tn + fp) == 0 ? 0.0 : (double) tn / (tn + fp);
    }

    public double f1() {
        double p = precision();
        double r = recall();
        return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
    }

    public double averageLoss() {
        return count == 0 ? 0.0 : totalLoss / count;
    }

    public void printSummary(String title) {
        System.out.println("\n================ " + title + " ================");
        System.out.println("Confusion Matrix (rows = actual, cols = predicted):");
        System.out.printf("%20s %10s %10s%n", "", "Pred 0", "Pred 1");
        System.out.printf("%20s %10d %10d%n", "Actual 0", tn, fp);
        System.out.printf("%20s %10d %10d%n", "Actual 1", fn, tp);
        System.out.println("-------------------------------------------------");
        System.out.printf("Test Samples : %d%n", count);
        System.out.printf("Avg Loss     : %.6f%n", averageLoss());
        System.out.printf("Accuracy     : %.4f (%.2f%%)%n", accuracy(), accuracy() * 100.0);
        System.out.printf("Precision    : %.4f%n", precision());
        System.out.printf("Recall       : %.4f%n", recall());
        System.out.printf("Specificity  : %.4f%n", specificity());
        System.out.printf("F1 Score     : %.4f%n", f1());
        System.out.println("=================================================");
    }
}
