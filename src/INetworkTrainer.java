public interface INetworkTrainer {
    INeuralNetwork train(INeuralNetwork network, Dataset samples);
}
