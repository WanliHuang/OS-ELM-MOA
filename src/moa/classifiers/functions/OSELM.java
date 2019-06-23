/*
 *    OSELM.java
 *    Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
 *    @author Wanli Huang (wanli.huang@gmail.com )
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.functions;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;
import no.uib.cipr.matrix.*;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Online Squential Extreme Learning Machine (OSELM) classifier.
 * For more details please refer to following paper
 * <br/>
 * <p>
 * &#64;article{huang2005line,
 * title={On-line sequential extreme learning machine.},
 * author={Huang, Guang-Bin and Liang, Nan-Ying and Rong, Hai-Jun and Saratchandran, Paramasivan and Sundararajan, Narasimhan},
 * journal={Computational Intelligence},
 * volume={2005},
 * pages={232--237},
 * year={2005},
 * publisher={Citeseer}
 * }
 * *
 * </p>
 *
 * <p>Performs classic perceptron multiclass learning incrementally.</p>
 *
 * <p>Parameters:</p>
 * <ul>
 * <li>-n : number of hidden neuron unit</li>
 * </ul>
 * <ul>
 * <li>-z : initial instances set size</li>
 * </ul>
 * <ul>
 * <li>-s :  seed to generate random value/li>
 * </ul>
 * <ul>
 * <li> -a : Set the activation function</li>
 * 1 - Sigmoid function
 * 2 - Sin function
 * 3 - Hardlim function
 * 4 - Yibas function
 * 5 - Radbas function
 * </ul>
 * <ul>
 * <li> -d : debug mode switch</li>
 * </ul>
 *
 * @author Wanli Huang (wanli.huang@gmail.com)
 * @version $Revision: 1 $
 */
public class OSELM extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 221L;

    @Override
    public String getPurposeString() {
        return "Online Sequential Extreme Learning Machine.";
    }

    public IntOption numHiddenNeuronsOption = new IntOption("HiddenNeuronNumber", 'n', "set the number of hidden neuron unit", 20);

    public IntOption numInitialSizeOption = new IntOption("DatasetIntialSize", 'z', "set the number of initial dataset size", 50);

    public IntOption isChunkFixedOption = new IntOption("isDataChunkFixed", 'c', "set if data chunk size is fixed or variable", 0);

    public IntOption minChunkSize = new IntOption("minChunkSize", 'i', "set the minimum chunk size", 1);

    public IntOption maxChunkSize = new IntOption("maxChunkSize", 'x', "set the maximum chunk size", 500);

    public IntOption randomSeedOption = new IntOption("RandomSeed", 's', "random seed", 1);

    public IntOption activationOption = new IntOption("ActivationFunction", 'a', "set activation function", 1);

    public IntOption simpleInverseOption = new IntOption("SimpleInverseMode", 'v', "set simple inverse mode on or off", 0);

    public IntOption normlizationOption = new IntOption("NormalizationMode", 'l', "set normalization mode on or off", 1);

    public IntOption debugOption = new IntOption("DebugMode", 'd', "set debug mode on or off", 0);


    protected DenseMatrix initialAttMatrix;

    protected DenseMatrix initialLabelMatrix;

    protected DenseMatrix initialInputWeightMatrix;

    protected DenseMatrix initialOutputWeightMatrix;

    protected DenseMatrix biases;

    protected DenseMatrix H0;

    protected DenseMatrix Kk;

    protected DenseMatrix Pk;

    protected DenseMatrix Hk;

    protected DenseMatrix Hk1;

    protected DenseMatrix Chunkk1;

    protected DenseMatrix Tk;

    protected DenseMatrix Tk1;

    protected DenseMatrix Bk;

    protected int chunkSize = 1;

    protected int k;

    protected boolean reset;

    protected int numberAttributes;

    protected int numberClasses;

    protected int classIndex;


    protected int initialColFlag;

    protected int chunkSizeFlag;

    protected double[][] m_normalization; //for hold min and max value for each attribute
    //protected int initialLabelColFlag;

    //protected String[] labels;
    protected double[] labels;

    protected double MAX;
    protected double MIN;

    //protected int attIndex;

    @Override
    public void resetLearningImpl() {
        this.initialColFlag = 0;
        this.chunkSizeFlag = 0;
        if (isChunkFixedOption.getValue() == 0) {
            chunkSize = getRandomNumberInRange(minChunkSize.getValue(), maxChunkSize.getValue());
        } else {
            chunkSize = isChunkFixedOption.getValue(); // fixed data chunk size. if data chunck size is 1, it means instance by instance
        }


        //initialAttColFlag = 0;
        this.reset = true;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        // Boosting Phrase
        //Initial training set , the number of instance is defined as numInitialSizeOption
        ;
        if (this.reset == true) {
            this.reset = false;

            this.numberAttributes = inst.numAttributes();
            this.numberClasses = inst.numClasses();
            this.classIndex = inst.classIndex();
            //this.labels = new String[this.numberClasses];
            this.labels = new double[this.numberClasses];
            this.initialAttMatrix = new DenseMatrix(this.numberAttributes - 1, this.numInitialSizeOption.getValue());
            this.initialLabelMatrix = new DenseMatrix(this.numberClasses, this.numInitialSizeOption.getValue());

            this.Chunkk1 = new DenseMatrix(this.numberAttributes - 1, this.chunkSize);
            this.Tk1 = new DenseMatrix(this.numberClasses, this.chunkSize);

            this.MAX = Double.MIN_VALUE;
            this.MIN = Double.MAX_VALUE;

            this.m_normalization = new double[2][this.numberAttributes - 1]; //
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < this.numberAttributes - 1; j++) {
                    if (i == 0) this.m_normalization[i][j] = this.MAX;
                    if (i == 1) this.m_normalization[i][j] = this.MIN;
                }
            }

            if (this.debugOption.getValue() == 1) {
                System.out.println("numberAttributes: " + this.numberAttributes);
                System.out.println("numberClasses: " + this.numberClasses);
                System.out.println("classIndex: " + this.classIndex);
            }




/*            Attribute classAtt = inst.classAttribute();

            for (int i = 0; i< this.numberClasses; i++){

                double v = Double.parseDouble(classAtt.value(i));
                this.labels[i] = v;
                if (this.debugOption.getValue() == 1){
                    System.out.println("class labels:" +this.labels[i]);
                }
            }*/

        }

        if (this.initialColFlag < this.numInitialSizeOption.getValue()) {


            int attIndex = 0; // index of attributeMatrix's row

            for (int i = 0; i < this.numberAttributes; i++) {

                if (i != this.classIndex) {


                    if (inst.value(i) > this.m_normalization[0][attIndex]) {
                        this.m_normalization[0][attIndex] = inst.value(i);
                    }

                    if (inst.value(i) < this.m_normalization[1][attIndex]) {
                        this.m_normalization[1][attIndex] = inst.value(i);
                    }


                    this.initialAttMatrix.set(attIndex, this.initialColFlag, inst.value(i));
                    if (this.debugOption.getValue() == 1) {
                        System.out.print("inst. attribute" + i + " value: " + inst.value(i) + " | ");
                    }


                    attIndex++;
                }
            }

            int label = (int) inst.value(this.classIndex);
            //String label = labelValue.toString();
            if (this.debugOption.getValue() == 1) {
                System.out.println("*");
                System.out.println("the label for this instance is： " + label);
            }

            for (int i = 0; i < this.numberClasses; i++) {
                this.initialLabelMatrix.set(i, this.initialColFlag, label == i ? 1 : -1); // fill all non-label with -1
            }


            this.initialColFlag++;

        } else if (this.initialColFlag == this.numInitialSizeOption.getValue()) {

            // Normalize the initial training dataset
            int rows = this.initialAttMatrix.numColumns(); // each instance
            int columns = this.initialAttMatrix.numRows(); // each attribute

            //normalization using x-min/(max-min)
            if (this.normlizationOption.getValue() == 1) {
                for (int j = 0; j < columns; j++) {

                    double min = m_normalization[1][j];
                    double max = m_normalization[0][j];
                    if (this.debugOption.getValue() == 1) {

                        System.out.print(j + " min: " + min + " ");
                        System.out.println(" max: " + max);

                    }

                    double normValue = 0;
                    for (int i = 0; i < rows; i++) {
                        if (max > min) {
                            normValue = (this.initialAttMatrix.get(j, i) - min) / (max - min);
                        } else if (max == min && max > 0) {
                            normValue = 1;
                        }
                        this.initialAttMatrix.set(j, i, normValue);
                    }


                }
            }


            if (this.debugOption.getValue() == 1) {
                printMX("Initial Attributes Matrix after normalization : ", this.initialAttMatrix);
            }

            if (this.debugOption.getValue() == 1) {
                printMX("Initial Label Matrix", this.initialLabelMatrix);
            }

            this.initialInputWeightMatrix = generateRandomMatrix(this.numHiddenNeuronsOption.getValue(), this.numberAttributes - 1, this.randomSeedOption.getValue());

            if (this.debugOption.getValue() == 1) {
                printMX("Initial Input Weight Matrix", this.initialInputWeightMatrix);
            }

            this.biases = generateRandomMatrix(this.numHiddenNeuronsOption.getValue(), 1, this.randomSeedOption.getValue());

            this.H0 = generateH(this.initialAttMatrix, this.initialInputWeightMatrix, this.biases, this.numInitialSizeOption.getValue());

            if (this.debugOption.getValue() == 1) {
                printMX("H0: ", H0);
            }

            DenseMatrix H0T = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.numInitialSizeOption.getValue());

            H0.transpose(H0T);

            DenseMatrix H0TH = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.numHiddenNeuronsOption.getValue());

            H0T.mult(H0, H0TH);

            this.Kk = H0TH;

            Inverse inverseOfHOTH = new Inverse(H0TH);

            this.Pk = inverseOfHOTH.getInverse();

            if (this.debugOption.getValue() == 1) {
                printMX("Initial Pk", this.Pk);
            }
            this.k = 0;
            //this.Hk = H0;
            //this.Tk = this.initialLabelMatrix;

            try {
                this.initialOutputWeightMatrix = generateOutputWeights(H0T, this.initialLabelMatrix, this.numInitialSizeOption.getValue());  // this is the target that the model is trained for
            } catch (Exception e) {

            }
            if (this.debugOption.getValue() == 1) {
                printMX("weightsOfOutput", this.initialOutputWeightMatrix);
            }

            this.Bk = this.initialOutputWeightMatrix;

            this.initialColFlag++;

        } else {


            //inst.

            //this.Hk1 = new DenseMatrix( this.k+1, this.numHiddenNeuronsOption.getValue());

            //this.;
            //Hk1.zero();


            if (this.chunkSizeFlag < this.chunkSize) {

                //System.out.println("This "+ k+1 +"th data chunk.");

                if (this.normlizationOption.getValue() == 1) {
                    int attIndex = 0; // index of attributeMatrix's row
                    // DeltaAttMatrix is to hold new instances in the data chunk
                    for (int i = 0; i < this.numberAttributes; i++) {

                        if (i != this.classIndex) {
                            //System.out.print(i+"th"+ " attribute value is" + inst.value(i)+ " ");
                            double min = this.m_normalization[1][attIndex]; // min value of ith attribute
                            double max = this.m_normalization[0][attIndex]; // max value of ith attribute

                            double normValue = 0;
                            if (inst.value(i) < min) {
                                this.m_normalization[1][attIndex] = inst.value(i); // set the min value of this attribute as this value
                            } else if (inst.value(i) > max) {
                                normValue = 1;
                                this.m_normalization[0][attIndex] = inst.value(i); // set the max value of this attribute as this value
                            } else {
                                if (max == min) {
                                    if (max > 0) normValue = 1;
                                } else if (max > min) {
                                    normValue = (inst.value(i) - min) / (max - min);
                                }
                            }
                            this.Chunkk1.set(attIndex, this.chunkSizeFlag, normValue);
                            attIndex++;
                        }
                    }

                } else {
                    int attIndex = 0; // index of attributeMatrix's row
                    // DeltaAttMatrix is to hold new instances in the data chunk
                    for (int i = 0; i < this.numberAttributes; i++) {

                        if (i != this.classIndex) {
                            this.Chunkk1.set(attIndex, this.chunkSizeFlag, inst.value(i));
                            attIndex++;
                        }
                    }
                }

                System.out.println("|");

                int label = (int) inst.value(this.classIndex);
                for (int i = 0; i < this.numberClasses; i++) {
                    this.Tk1.set(i, this.chunkSizeFlag, label == i ? 1 : -1); // fill all non-label with -1
                }

                this.chunkSizeFlag++;


            } else if (this.chunkSizeFlag == this.chunkSize) {


                if (this.debugOption.getValue() == 1) {
                    printMX("Data Chunk Matrix: ", Chunkk1);
                }

                if (this.debugOption.getValue() == 1) {
                    printMX("Data Chunk Label Matrix: ", Tk1);
                }

                DenseMatrix Hk1 = generateH(this.Chunkk1, this.initialInputWeightMatrix, this.biases, this.chunkSize);

                if (this.debugOption.getValue() == 1) {
                    printMX("Hk1: ", Hk1);
                }

                DenseMatrix Hk1T = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.chunkSize);

                Hk1.transpose(Hk1T);

                //DenseMatrix Hk1THk1 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(),this.numHiddenNeuronsOption.getValue());

                //Hk1T.mult(Hk1, Hk1THk1);

                //Inverse inverseOfHk1THk1 = new Inverse(Hk1THk1);

                //this.Pk = inverseOfHk1THk1.getInverse();
                DenseMatrix Hk1Pk = new DenseMatrix(this.chunkSize, this.numHiddenNeuronsOption.getValue());


                Hk1.mult(this.Pk, Hk1Pk);

                DenseMatrix Hk1PkHk1T = new DenseMatrix(this.chunkSize, this.chunkSize);

                Hk1Pk.mult(Hk1T, Hk1PkHk1T);

                DenseMatrix I = Matrices.identity(this.chunkSize);

                Hk1PkHk1T.add(I);

                Inverse inverse = new Inverse(Hk1PkHk1T);


                DenseMatrix inverseOfHk1PkHk1T = new DenseMatrix(this.chunkSize, this.chunkSize);
                ;
                inverseOfHk1PkHk1T = inverse.getInverse();

                if (this.debugOption.getValue() == 1) {
                    printMX("the inverse of Hk1PkHk1T: ", inverseOfHk1PkHk1T);
                }

                DenseMatrix MX1 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.chunkSize);
                this.Pk.mult(Hk1T, MX1);
                if (this.debugOption.getValue() == 1) {
                    printMX("PkHk1T: ", MX1);
                }

                DenseMatrix MX2 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.chunkSize);

                MX1.mult(inverseOfHk1PkHk1T, MX2);

                if (this.debugOption.getValue() == 1) {
                    printMX("PkHk1T * iverseoOfHk1PkHk1T: ", MX2);
                }

                DenseMatrix MX3 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.numHiddenNeuronsOption.getValue());

                MX2.mult(Hk1, MX3);

                if (this.debugOption.getValue() == 1) {
                    printMX("PkHk1T * iverseoOfHk1PkHk1T * Hk1: ", MX3);
                }

                DenseMatrix MX4 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.numHiddenNeuronsOption.getValue());

                MX3.mult(this.Pk, MX4);

                if (this.debugOption.getValue() == 1) {
                    printMX("PkHk1T * iverseoOfHk1PkHk1T * Hk1Pk: ", MX4);
                }

                this.Pk.add(-1, MX4);

                if (this.debugOption.getValue() == 1) {
                    printMX("Pk1: ", Pk);
                }

                //*****************************************************

                DenseMatrix tempMX1 = new DenseMatrix(this.chunkSize, numberClasses);
                Hk1.mult(this.Bk, tempMX1);

                if (this.debugOption.getValue() == 1) {
                    printMX("Hk1* Bk: ", tempMX1);
                }
                DenseMatrix Tk1T = new DenseMatrix(this.chunkSize, numberClasses);
                this.Tk1.transpose(Tk1T);

                Tk1T.add(-1, tempMX1);

                if (this.debugOption.getValue() == 1) {
                    printMX("Tk1-Hk1* Bk: ", this.Tk1);
                }


                DenseMatrix tempMX2 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.chunkSize);

                Pk.mult(Hk1T, tempMX2);

                if (this.debugOption.getValue() == 1) {
                    printMX("Pk1 * Hk1T: ", tempMX2);
                }

                DenseMatrix tempMX3 = new DenseMatrix(this.numHiddenNeuronsOption.getValue(), this.numberClasses);

                tempMX2.mult(Tk1T, tempMX3);

                if (this.debugOption.getValue() == 1) {
                    printMX("Pk1 * Hk1T * (Tk1 - Hk1Bk): ", tempMX3);
                }


                this.Bk.add(tempMX3);


                if (this.debugOption.getValue() == 1) {
                    printMX("BK1: ", this.Bk);
                }

                //this.Bk1 = Bk;

                if (this.debugOption.getValue() == 1) {
                    printMX("Bk+Pk1 * Hk1T * (Tk1 - Hk1Bk) - updated Bk1: ", this.Bk);
                }

                this.chunkSizeFlag = 0; // reset chunk size for next data chunk

                if (isChunkFixedOption.getValue() == 0) {
                    this.chunkSize = getRandomNumberInRange(minChunkSize.getValue(), maxChunkSize.getValue());
                    this.Chunkk1 = new DenseMatrix(this.numberAttributes - 1, this.chunkSize);
                    this.Tk1 = new DenseMatrix(this.numberClasses, this.chunkSize);
                }

                //this.k++;  // next data chunk

            }

        }


    }


    public int getNumberAttributes() {

        return this.numberAttributes;
    }

    public int getNumberClasses() {

        return this.numberClasses;
    }


    @Override
    public double[] getVotesForInstance(Instance inst) {

        int numAttributes = inst.numAttributes();
        int clsIndex = inst.classIndex();
        int numTestAtt = numAttributes - 1;

        //double[] testData = new double[numTestAtt];

        double[] votes = new double[inst.numClasses()];

        if (this.initialColFlag > this.numInitialSizeOption.getValue()) {

            DenseMatrix testData = new DenseMatrix(numTestAtt, 1);
            DenseMatrix output = new DenseMatrix(1, this.numberClasses);


            if (this.reset == false) {
                int attIndex = 0; // index of attributeMatrix's row
                for (int i = 0; i < numAttributes; i++) {

                    if (i != clsIndex) {

                        //System.out.print(i+"th"+ " attribute value is" + inst.value(i)+ " ");
                        double min = this.m_normalization[1][attIndex]; // min value of ith attribute
                        double max = this.m_normalization[0][attIndex]; // max value of ith attribute

                        double normValue = 0;
                        if (inst.value(i) > max) {
                            normValue = 1;
                        } else if (inst.value(i) < min) {
                            normValue = 0;
                        } else if (max == min && max > 0 && inst.value(i) == max) {
                            normValue = 1;
                        } else if (max == min && max <= 0 && inst.value(i) == max) {
                            normValue = 0;
                        } else if (max > min) {
                            normValue = (inst.value(i) - min) / (max - min);
                        }


                        testData.set(attIndex, 0, normValue);

                        attIndex++;

                    }
                }


                DenseMatrix H_test = generateH(testData, this.initialInputWeightMatrix, this.biases, 1);

                DenseMatrix H_test_T = new DenseMatrix(1, this.numHiddenNeuronsOption.getValue());

                // H_test.transpose(H_test_T);


                //H_test_T.mult(this.Bk, output);
                H_test.mult(this.Bk, output);

            }

            for (int i = 0; i < this.numberClasses; i++) {
                votes[i] = output.get(0, i);
            }

/*            try {
                Utils.normalize(votes);
            } catch (Exception var4) {
                ;
            }*/

            for (int i = 0; i < this.numberClasses; i++) {
                System.out.print(votes[i] + " ");
            }
            System.out.println("|");
        }
        return votes;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList();
        return (Measurement[]) measurementList.toArray(new Measurement[measurementList.size()]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    /**
     * @param AttrMatrix
     * @param InputWeightsMatrix
     * @param Biases
     * @param numOfInstances
     * @return a matrix containing G(w*x + b)  dimension Row:  number of instances  Column: number of hidden neutrons
     */

    private DenseMatrix generateH(DenseMatrix AttrMatrix, DenseMatrix InputWeightsMatrix, DenseMatrix Biases, int numOfInstances) {

        int m_numHiddenNeurons = this.numHiddenNeuronsOption.getValue();
        DenseMatrix tempH = new DenseMatrix(m_numHiddenNeurons, numOfInstances);
        InputWeightsMatrix.mult(AttrMatrix, tempH);
        DenseMatrix BiasMX = new DenseMatrix(m_numHiddenNeurons, numOfInstances);

        for (int i = 0; i < numOfInstances; i++) {
            for (int j = 0; j < m_numHiddenNeurons; j++) {
                BiasMX.set(j, i, Biases.get(j, 0));   // fill up each column (instance) with bias value;
            }
        }


        tempH.add(BiasMX);

        //DenseMatrix H = new DenseMatrix(m_numHiddenNeurons, numOfInstances);


        for (int i = 0; i < m_numHiddenNeurons; i++) {
            for (int j = 0; j < numOfInstances; j++) {
                double v = Activation(tempH.get(i, j), this.activationOption.getValue());
                tempH.set(i, j, v);
            }
        }

        DenseMatrix H = new DenseMatrix(numOfInstances, m_numHiddenNeurons);
        tempH.transpose(H);

        return H;


    }

    /**
     * generate output weight matrix
     *
     * @param H
     * @param classesMatrix
     * @return a matrix containing Output weights  dimension Row: number of hidden neutrons Column: number of output neutrons
     */
    private DenseMatrix generateOutputWeights(DenseMatrix H, DenseMatrix classesMatrix, int numInstances) throws NotConvergedException {

        int m_numHiddenNeurons = this.numHiddenNeuronsOption.getValue();
        int m_numOfOutputNeutrons = this.numberClasses;
        DenseMatrix HT = new DenseMatrix(numInstances, m_numHiddenNeurons);
        DenseMatrix InvHT = new DenseMatrix(m_numHiddenNeurons, numInstances);
        H.transpose(HT);
        Inverse inverseOfHT = new Inverse(HT, this.randomSeedOption.getValue());
        if (m_numHiddenNeurons == numInstances && simpleInverseOption.getValue() == 1) {
            InvHT = inverseOfHT.getInverse();
        } else {
            InvHT = inverseOfHT.getMPInverse();
        }
        if (this.debugOption.getValue() == 1) {
            printMX("MoorePenroseInvHT / InvHT", InvHT);
        }
        DenseMatrix outputWeightsMX = new DenseMatrix(m_numHiddenNeurons, m_numOfOutputNeutrons);

        DenseMatrix TransposedClassesMX = new DenseMatrix(numInstances, m_numOfOutputNeutrons);

        classesMatrix.transpose(TransposedClassesMX);


        InvHT.mult(TransposedClassesMX, outputWeightsMX);

        return outputWeightsMX;

    }


    /**
     * Different activation functions
     *
     * @param value
     * @param Activation_type
     * @return return activation function's result
     */

    private double Activation(double value, int Activation_type) {

        double result = 0.0;
        if (Activation_type == 1) {    //Sig

            result = 1.0f / (1 + Math.exp(-value));

        } else if (Activation_type == 2) {  //Sin

            result = Math.sin(value);

        } else if (Activation_type == 3) { //Hardlim
            // to do
        } else if (Activation_type == 4) { //Yibas
            // to do
        } else if (Activation_type == 5) { //Radbas
            double a = 2, b = 2, c = Math.sqrt(2);
            result = a * Math.exp(-(value - b) * (value - b) / c * c);
        }
        return result;
    }

    /**
     * Generate a matrix with random values
     *
     * @param rows
     * @param cols
     * @param m_seed
     * @return return activation function's result
     */
    private static DenseMatrix generateRandomMatrix(int rows, int cols, int m_seed) {

        Random random_value = new Random(m_seed);
        DenseMatrix randomMX = new DenseMatrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                randomMX.set(i, j, random_value.nextDouble());
            }
        }
        return randomMX;
    }

    private void printMX(String nameMX, DenseMatrix MX) {

        System.out.println(nameMX + "numRows: " + MX.numRows());
        System.out.println(nameMX + "numColumns: " + MX.numColumns());
        for (int i = 0; i < MX.numRows(); i++) {
            for (int j = 0; j < MX.numColumns(); j++) {
                System.out.print(MX.get(i, j) + ", ");
            }
            System.out.println("//");
        }

        System.out.println("Press Enter to continue");
        try {
            System.in.read();
        } catch (Exception e) {
        }

    }

    private static int getRandomNumberInRange(int min, int max) {

        if (min >= max) {
            throw new IllegalArgumentException("max must be greater than min");
        }

        Random r = new Random();
        return r.nextInt((max - min) + 1) + min;
    }


}
