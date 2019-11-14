public class Main {

    public  static void main(String[] args)
    {
//        J48 j48 = new J48();
//        String defaultJ48 = "-C 0.25 -M 2";
//        j48.run(defaultJ48);

        RandomForest randomForest = new RandomForest();
        String defaultRF = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1";
        randomForest.run(defaultRF);
    }

}
