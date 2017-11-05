import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Dataset extends ArrayList<Sample> {

    public Dataset() {

    }

    public Dataset(List<Sample> samples) {
        this.addAll(samples);
    }

    public void shuffle() {
        Collections.shuffle(this);
    }

    public Dataset getTrainingSet() {
        return new Dataset(this.subList(0, this.size() / 2));
    }

    public Dataset getTestingSet() {
        return new Dataset(this.subList(this.size() / 2, this.size()));
    }
}
