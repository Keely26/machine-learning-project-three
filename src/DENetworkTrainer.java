import java.util.List;

public class DENetworkTrainer extends NetworkTrainerBase {

    DENetworkTrainer() {

    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return network;
    }
}
