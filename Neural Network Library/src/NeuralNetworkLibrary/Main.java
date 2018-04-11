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
        NeuralNetwork nn = new NeuralNetwork(2, 1, 3);
        double[][] values = new double[2][2];
        values[0][0] = 40;
        values[0][1] = 10;
        values[1][0] = 2;
        values[1][1] = 5;
        System.out.println(Arrays.deepToString(nn.forward(values)));
    }
    
}
