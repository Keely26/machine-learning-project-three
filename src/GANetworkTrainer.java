import java.util.List;

public class GANetworkTrainer extends NetworkTrainerBase {

    GANetworkTrainer() {

    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return network;
    }
}
