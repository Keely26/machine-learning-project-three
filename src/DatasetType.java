public enum DatasetType {
    Ecoli("ecoli"),
    Energy("energy"),
    TicTacToe("tic-tac-toe"),
    Wine("winequality"),
    Yeast("yeast");

    private String fileName;

    DatasetType(String fileName) {
        this.fileName = fileName;
    }

    @Override
    public String toString() {
        return fileName;
    }
}
