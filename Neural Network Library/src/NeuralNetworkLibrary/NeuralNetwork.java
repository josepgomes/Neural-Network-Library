package NeuralNetworkLibrary;

import java.util.Random;

/**
 *
 * @author josepgomes
 */
public class NeuralNetwork {
    
    double[][] w1;
    double[][] w2;
    
    public NeuralNetwork(int inputLayerSize, int outputLayerSize, int hiddenLayerSize){
        w1 = new double[inputLayerSize][hiddenLayerSize];
        w2 = new double[hiddenLayerSize][outputLayerSize];
        
        Random random = new Random();
        
        for (int i = 0; i < inputLayerSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                w1[i][j] = random.nextGaussian();
            }
        }
        
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < outputLayerSize; j++) {
                w2[i][j] = random.nextGaussian();
            }
        }
    }
    
    public double[][] forward(double[][] values){
        double[][] z2 = dot(values, w1);
        double[][] a2 = sigmoid(z2, false);
        double[][] z3 = dot(a2, w2);
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
