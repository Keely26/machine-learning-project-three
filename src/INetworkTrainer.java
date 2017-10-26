import java.util.List;

public interface INetworkTrainer {
    INeuralNetwork train(INeuralNetwork network, List<Sample> samples);
}
