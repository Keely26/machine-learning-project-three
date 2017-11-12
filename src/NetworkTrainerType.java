enum NetworkTrainerType {

    BPNetworkTrainer("Backpropagation Trainer"),
    ESNetworkTrainer("Evolution Strategy Trainer"),
    DENetworkTrainer("Differential Evolution Trainer"),
    GANetworkTrainer("Genetic Algorithm Trainer");

    private final String trainerName;

    NetworkTrainerType(String trainerName) {
        this.trainerName = trainerName;
    }

    @Override
    public String toString() {
        return this.trainerName;
    }
}
