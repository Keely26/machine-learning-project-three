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

    public Dataset trainSet() {
        return new Dataset(this.subList(0, this.size() / 2));
    }

    public Dataset testSet() {
        return new Dataset(this.subList(this.size() / 2, this.size()));
    }
}
