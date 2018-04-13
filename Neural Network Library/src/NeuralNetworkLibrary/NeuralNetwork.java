package NeuralNetworkLibrary;

import java.util.Random;

/**
 *
 * @author josepgomes
 */
public class NeuralNetwork {

    double[][][] weights;

    public NeuralNetwork(int inputLayerSize, int outputLayerSize, int hiddenLayers, int hiddenLayerSize, boolean randomWeights) {
        weights = new double[hiddenLayers + 1][][];
        weights[0] = new double[inputLayerSize][hiddenLayerSize];
        weights[weights.length - 1] = new double[hiddenLayerSize][outputLayerSize];

        for (int i = 1; i < weights.length - 1; i++) {
            weights[i] = new double[hiddenLayerSize][hiddenLayerSize];;
        }

        if (randomWeights) {
            Random random = new Random();

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int n = 0; n < weights[i][j].length; n++) {
                        weights[i][j][n] = random.nextGaussian();
                    }
                }
            }
        }
    }

    public void setWeights(double[][][] weights) {
        this.weights = weights;
    }

    public double[][] forward(double[][] values) {
        double[][] z2 = dot(values, weights[0]);
        double[][] a2 = sigmoid(z2, false);
        double[][] ax = a2.clone();
        for (int i = 1; i < weights.length - 1; i++) {
            double[][] zx = dot(ax, weights[i]);
            ax = sigmoid(zx, false);
        }
        double[][] z3 = dot(ax, weights[weights.length - 1]);
        double[][] result = sigmoid(z3, false);
        return result;
    }

    public double sigmoid(double t) {
        return 1 / (1 + Math.pow(Math.E, (-1 * t)));
    }

    public double[][] sigmoid(double[][] x, boolean deriv) {
        double[][] result = new double[x.length][x[0].length];

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                double sigmoidCell = sigmoid(x[i][j]);

                if (deriv == true) {
                    result[i][j] = sigmoidCell * (1 - sigmoidCell);
                } else {
                    result[i][j] = sigmoidCell;
                }
            }
        }

        return result;
    }

    public double[][] dot(double[][] A, double[][] B) {
        int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = B[0].length;

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }

        double[][] C = new double[aRows][bColumns];
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bColumns; j++) {
                C[i][j] = 0.00000;
            }
        }

        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bColumns; j++) {
                for (int k = 0; k < aColumns; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

}
