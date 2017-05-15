import weka.core.converters.ConverterUtils.DataSource;

public class Interface{


    private ClassifyData classify;

    /**
     * Launch the application.
     */
    public void main(String[] args) {
        //TODO: pass location by arg
        try {
            classify = new ClassifyData();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }
}
