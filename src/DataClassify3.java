import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataClassify3 {


    static MultilayerPerceptron classifier;
    static Evaluation evaluation;
    static Instances fileInstances, train, test;

    public DataClassify3() throws Exception {
        classifier = new MultilayerPerceptron();
        classifier.setOptions(Utils.splitOptions("-L 0.4 -M 0.2 -N 700 -V 0 -S 0 -E 20 -H 17"));
    }

    public void test() throws Exception {
        test.setClassIndex(train.numAttributes() - 1);

        evaluation = new Evaluation(train);
        evaluation.evaluateModel(classifier, test);
        System.out.println("Numero de casos: " + evaluation.numInstances());
        System.out.println("Corrects: " + evaluation.correct());
        System.out.println("Incorrectos: " + evaluation.incorrect());
        System.out.println("percentagem de erro: " + evaluation.errorRate());
        System.out.println(evaluation.toMatrixString());

    }

    public void train() throws Exception {
        train.setClassIndex(train.numAttributes() - 1);
        classifier.buildClassifier(train);
    }

    public String classifyProduct(ArrayList<Double> params) throws Exception {

        ArrayList<String> returnValues = new ArrayList<String>(2);
        returnValues.add("RB");
        returnValues.add("NRB");

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();

        for (int i = 0; i < params.size(); i++) {

            Attribute temp = new Attribute((i + 1) + "");

            attributes.add(temp);
        }

        Attribute lastAttr = new Attribute("@@class@@", returnValues);
        attributes.add(lastAttr);
        ArrayList<Attribute> atts = new ArrayList<Attribute>(2);

        Instances newCases = new Instances("newCase", attributes, 1);
        Instance newCaseInstance = new DenseInstance(newCases.numAttributes());
        for (int i = 0; i < 40; i++) {
            newCaseInstance.setValue(attributes.get(i), params.get(i));
        }
        newCases.add(newCaseInstance);
        newCases.setClassIndex(newCases.numAttributes() - 1);


        Instances labeled = new Instances(newCases);

        double label = classifier.classifyInstance(newCases.instance(0));
        labeled.instance(0).setClassValue(label);

        return newCases.classAttribute().value((int) label);
    }

    public void filesOpen(BufferedReader file, int percentage) throws IOException {

        fileInstances = new Instances(file);
        int trainSize = (int) Math.round(fileInstances.numInstances() * percentage / 100);
        int testSize = fileInstances.numInstances() - trainSize;
        train = new Instances(fileInstances, 0, trainSize);
        test = new Instances(fileInstances, trainSize, testSize);


    }

    public double getNumEval() {
        return evaluation.numInstances();
    }

    public String getMatrix() throws Exception {
        return evaluation.toMatrixString("");

    }

    public double getCorrect() throws Exception {
        return evaluation.correct();
    }

    public double getIncorrect() throws Exception {
        return evaluation.incorrect();
    }

    public double getPercentage() throws Exception {
        return evaluation.errorRate();
    }
}
