import weka.core.converters.ConverterUtils.DataSource;

public class Interface{



    /**
     * Launch the application.
     */
    public static void main(String[] args) {
        //TODO: pass location by arg
         ClassifyData classify = null;


        try {
            classify = new ClassifyData();
        } catch (Exception e) {
            e.printStackTrace();
        }
        classify.setInstances(args[0],Integer.parseInt(args[1]));
        try {
            classify.train();
            classify.test();
            classify.save_tree();
        } catch (Exception e) {
            e.printStackTrace();
        }
        //input instance

    }
}
