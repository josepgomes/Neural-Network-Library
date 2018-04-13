package NeuralNetworkLibrary;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author josepgomes
 */
public class Trainer {

    private static final double crossoverRate = 0.5;
    private static final double mutationRate = 0.015;
    private static final int tournamentSize = 5;
    private static final boolean elitism = true;
    private NeuralNetwork[] population;
    private double[] populationFitness;
    private int inputLayerSize;
    private int outputLayerSize;
    private int hiddenLayers;
    private int hiddenLayerSize;

    public Trainer(int populationSize, int inputLayerSize, int outputLayerSize, int hiddenLayers, int hiddenLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayers = hiddenLayers;
        this.hiddenLayerSize = hiddenLayerSize;
        population = new NeuralNetwork[populationSize];
        populationFitness = new double[populationSize];

        generatePopulation();
    }

    public void generatePopulation() {
        for (int i = 0; i < population.length; i++) {
            population[i] = new NeuralNetwork(inputLayerSize, outputLayerSize, hiddenLayers, hiddenLayerSize, true);
        }
    }

    //custom function
    public double calculateFitness(int index) {
        double[][] values = new double[4][2];
        values[0][0] = 1;
        values[0][1] = 1;
        values[1][0] = 1;
        values[1][1] = 0;
        values[2][0] = 0;
        values[2][1] = 1;
        values[3][0] = 0;
        values[3][1] = 0;

        double[][] results = population[index].forward(values);
        double error = Math.abs(0 - results[0][0])
                + Math.abs(1 - results[1][0])
                + Math.abs(1 - results[2][0])
                + Math.abs(0 - results[3][0]);
        double normalisedError = 40 - (error * 10);

        populationFitness[index] = normalisedError;
        
        return normalisedError;
    }

    public NeuralNetwork crossover(int index1, int index2) {
        double[][][] index1Weights = population[index1].weights.clone();
        double[][][] index2Weights = population[index2].weights.clone();
        double[][][] newWeights = population[index1].weights.clone();

        for (int i = 0; i < index1Weights.length; i++) {
            for (int j = 0; j < index1Weights[i].length; j++) {
                for (int n = 0; n < index1Weights[i][j].length; n++) {
                    if (Math.random() <= crossoverRate) {
                        newWeights[i][j][n] = index1Weights[i][j][n];
                    } else {
                        newWeights[i][j][n] = index2Weights[i][j][n];
                    }
                }
            }
        }

        NeuralNetwork newNN = new NeuralNetwork(inputLayerSize, outputLayerSize, hiddenLayers, hiddenLayerSize, false);
        newNN.setWeights(newWeights);

        return newNN;
    }

    public void mutate(int index) {
        double[][][] weights = population[index].weights.clone();
        double[][][] newWeights = population[index].weights.clone();
        
        Random random = new Random();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int n = 0; n < weights[i][j].length; n++) {
                    if (Math.random() <= mutationRate) {
                        newWeights[i][j][n] = random.nextGaussian();
                    }
                }
            }
        }
        
        population[index].setWeights(newWeights);
    }

    public void tournament() {
        
    }
}
