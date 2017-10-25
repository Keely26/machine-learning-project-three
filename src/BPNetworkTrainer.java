import java.util.List;

public class BPNetworkTrainer extends NetworkTrainerBase {

    private double learningRate;
    private double momentum;
    private int batchSize;
    private int epochs;

    BPNetworkTrainer(double learningRate, double momentum, int batchSize, int epochs) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.batchSize = batchSize;
        this.epochs = epochs;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return network;
    }
}
