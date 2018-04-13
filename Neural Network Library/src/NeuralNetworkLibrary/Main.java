package NeuralNetworkLibrary;

import java.util.Arrays;

/**
 *
 * @author josepgomes
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        double[][] values = new double[4][2];

        values[0][0] = 1;
        values[0][1] = 1;

        values[1][0] = 1;
        values[1][1] = 0;

        values[2][0] = 0;
        values[2][1] = 1;

        values[3][0] = 0;
        values[3][1] = 0;

        NeuralNetwork nn = new NeuralNetwork(2, 1, 5, 3);
        System.out.println(Arrays.deepToString(nn.forward(values)));

    }

}
