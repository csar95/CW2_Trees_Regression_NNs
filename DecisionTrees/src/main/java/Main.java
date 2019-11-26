public class Main {

    public  static void main(String[] args)
    {
        J48 j48 = new J48();
        String[] settings = {
                "-C 0.25 -M 2",
                "-C 0.25 -M 4",
                "-C 0.25 -B -M 2",
                "-C 0.25 -B -M 4",
                "-C 0.5 -M 2",
                "-C 0.5 -M 4",
                "-C 0.5 -B -M 2",
                "-C 0.5 -B -M 4",
                "-U -M 2",
                "-U -M 4"
        };
        for (String s : settings) {
            for (Datasets d : Datasets.values()) {
                j48.run(s, d);
            }
        }

//        j48.run("-C 0.75 -M 2", Datasets.KFOLD);

//        RandomForest randomForest = new RandomForest();
//        String defaultRF = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1";
//        randomForest.run(defaultRF);

//        UserClassifier userClassifier = new UserClassifier();
//        userClassifier.run();
    }

}