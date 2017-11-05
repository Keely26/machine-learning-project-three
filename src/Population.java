import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Population extends ArrayList<WeightMatrix> {

    public Population(List<WeightMatrix> individuals) {
        this.addAll(individuals);
    }

    public Population() {

    }

    /**
     * Compare the fitness of each individual in the population, returning argmax(greatest)
     */
    public WeightMatrix getMostFit() {
        WeightMatrix mostFit = this.get(0);

        for (WeightMatrix individual : this) {
            if (individual.getFitness() > mostFit.getFitness()) {
                mostFit = individual;
            }
        }

        return mostFit;
    }

    /**
     * Reorder the population according to fitness, most fit first
     */
    public void sortByFitness() {
        this.sort(Comparator.comparingDouble(WeightMatrix::getFitness));
    }
}
