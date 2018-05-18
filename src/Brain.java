import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import org.ejml.simple.SimpleMatrix;

public abstract class Brain {
	
	public static final double XAVIER_NORMALIZED_INIT = Math.sqrt(6);
	public static final int MAX_TRAINING_ITERATIONS = 50;
	
	private int numLabels;
	private int[] layerSizes;
	protected String[] dataFileLines;
	protected TestCase[] cases;
	protected TestCase[] trainingSet;
	protected TestCase[] validationSet;
	protected TestCase[] testSet;
	protected String[] labels; // All the possible output values of the brain
	private double[][][] thetas;
	private Random rand; // Random number generator
	private double lambda; // Weight regularization parameter
	
	private static double[][][] gradients; // Most recent gradients calculated by costFunction
	
	/**
	 * Initializes the brain.
	 * @param layerSizes The number of nodes in each layer of the network. Index 0 is the input layer, and the last number is the output layer.
	 * @throws BrainException
	 */
	public Brain(int[] layerSizes) throws BrainException {
		
		if (layerSizes.length >= 2) {
			
			this.layerSizes = layerSizes;
			numLabels = layerSizes[layerSizes.length - 1]; // output layer
			labels = new String[numLabels];
			for (int x = 0 ; x < numLabels ; x++) // Set default labels to "0", "1", "2", etc
				labels[x] = x + "";
			
			// Randomly initialize NN weights
			rand = new Random();
			thetas = randInitializeNNWeights();
			lambda = 1;
			
		} else {
			
			throw new BrainException("Too few layers for a working brain! Please make at least two layers: one for input and one for output.");
			
		}
	}
	
	/**
	 * Now that the brain has been constructed, it can now load in a data file & its contents.
	 * @param information Relevant information to load data, including file name and directory.
	 * @throws BrainException 
	 */
	public abstract void loadData(String[] information) throws BrainException;
	
	/**
	 * Trains the neural network after data is loaded into test case sets.
	 */
	public void train() {
		
		// Set casesX as a matrix of inputs with a bias column of 1s
		double[][] casesX = new double[cases.length][layerSizes[0]];
		for (int i = 0 ; i < cases.length ; i++) {
			
			casesX[i][0] = 1; // bias column
			
			for (int j = 0 ; j < layerSizes[0] ; j++) {
				
				casesX[i][j + 1] = cases[i].getInput()[j];
				
			}
			
		}
		
		// Recode known answers as an array of vectors with 1 at the index of the y value, 0 at all other indices:
		double[][] casesY = new double[cases.length][layerSizes[layerSizes.length - 1]];
		for (int i = 0 ; i < cases.length ; i++)
			casesY[i][cases[i].getExpectedOutput()] = 1;
		
		// Convert into SimpleMatrix objects for efficient matrix computation
		SimpleMatrix x = new SimpleMatrix(casesX);
		SimpleMatrix y = new SimpleMatrix(casesY);
		
		thetas = fmincg(thetas , MAX_TRAINING_ITERATIONS);
		
	}
	
	/**
	 * 
	 * @param thetas
	 * @param maxIterations
	 * @return
	 */
	private static double[][][] fmincg(double[][][] thetas , int maxIterations) {
		
		
		
	}
	
	/**
	 * 
	 * @return the cost J; sets the static class variable 'gradients' to the gradients calculated
	 */
	private static double costFunction(double[][][] thetaMatrices , int[] layerSizes , double[][] casesX , double[][] casesY , double lambda) {
		
		int m = casesX.length;
		double J = 0;
		int numLabels = layerSizes[layerSizes.length - 1];
		
		// Convert thetas into SimpleMatrix objects for efficient matrix computation
		SimpleMatrix[] thetas = new SimpleMatrix[thetaMatrices.length];
		for (int i = 0 ; i < thetaMatrices.length ; i++)
			thetas[i] = new SimpleMatrix(thetaMatrices[i]);
		
		SimpleMatrix a = new SimpleMatrix(casesX);
		// Feedforward through the layers:
		for (int i = 0 ; i < thetas.length ; i++) {
			
			// Feedforward one layer
			a = sigmoid(a.mult(thetas[i].transpose()));
			
			// If not the last feedforward step, then append a bias column of 1s
			if (i + 1 < thetas.length) {
				
				// Workaround of SimpleMatrix limitation of no left-growing:
				//  (0 , 0)		(1 , 0)		(1  , a00, .. , a0j)
				//	(0 , 0)	->	(1 , 0)	->	(1  , a10, .. , a1j)
				//	(. , .)		(. , .)		(.  , .  , .. , .  )
				//	(0 , 0)		(1 , 0)		(1  , ai0, .. , aij)
				a = (new SimpleMatrix(a.numRows() , 2))
						.combine(0 , 0 , new SimpleMatrix(a.numRows() , 1).plus(1))
						.combine(0 , 1 , a);
				
			}
			
		}
		
		// Compute J:
		for (int i = 0 ; i < numLabels ; i++) {
			
			J = J + (1.0 / m) * 
			
		}
		
		// Regularize J:
		double thetasSquaredSum = 0;
		for (SimpleMatrix mx : thetas)
			thetasSquaredSum += mx.elementPower(2).elementSum();
		J = J + (lambda / (2.0 * m)) * thetasSquaredSum;
		
	}
	
	/**
	 * Calculates the sigmoid equation for each element in a SimpleMatrix and returns the result.
	 * @param mx
	 * @return
	 */
	public static SimpleMatrix sigmoid(SimpleMatrix mx) {
		
		// Equivalent to 1 / (1 + e^(mx)) for each element in mx
		return mx.scale(-1).elementExp().plus(1).elementPower(-1);
		
	}

	/**
	 * Reads a file and returns its contents in an array of lines of text.
	 * @param fname The file path + name.
	 * @return An array with each entry a line of text from the file.
	 */
	protected String[] fileToLines(String fname) throws BrainException {
		
		String[] content = new String[100];
        String line = null;

        try {
        	
            FileReader fileReader = new FileReader(fname);

            // Wrap the FileReader in a BufferedReader.
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            int index = 0;
            while((line = bufferedReader.readLine()) != null) {
            	
            	// Dynamically increase array size if going to overflow
                if (content.length <= index) {
                	
                	String[] temp = content;
                	content = new String[content.length * 2];
                	System.arraycopy(temp , 0 , content , 0 , temp.length);
                	
                }
                
                content[index] = line;
                index++;
                
            }

            // Close the file
            bufferedReader.close();
            
            // Reduce the array size to not have any empty trailing entries
            
            
            return content;
            
        }
        catch(FileNotFoundException ex) {
        	
            throw new BrainException("The brain can't work with an invalid data file path!");
            
        }
        catch(IOException ex) {
        	
            throw new BrainException("The brain hiccupped reading the data path...");
            
        }
		
	}
	
	/**
	 * Splits all the cases provided into training set, validation set, and test set 50/25/25
	 */
	protected void splitCases() {
		
		int numCases = cases.length;
		int trainingSplit = (int) (numCases / 2.0);
		int validationSplit = (int) (trainingSplit + numCases / 4.0);
		
		trainingSet = Arrays.copyOfRange(cases , 0 , trainingSplit);
		validationSet = Arrays.copyOfRange(cases , trainingSplit , validationSplit);
		testSet = Arrays.copyOfRange(cases , validationSplit , cases.length);
		
	}
	
	/**
	 * Creates a 3D array of randomly initialized weights for every NN connection specified by the layer sizes.
	 * @return The 3D array of randomly initialized weights.
	 */
	private double[][][] randInitializeNNWeights() {
		
		double[][][] result = new double[layerSizes.length - 1][][];
		for (int x = 0 ; x < layerSizes.length - 1 ; x++) {
			
			result[x] = randInitializeWeights(layerSizes[x] , layerSizes[x + 1]);
			
		}
		
		return result;
		
	}
	
	/**
	 * Creates a 2D array of randomly initialized weights (theta) for the connections between layers of specified sizes, in->out.
	 * @param in The size of the input layer of the connection.
	 * @param out The size of the output layer of the connection.
	 * @return The 2D array of randomly initialized weights (theta) for the connections between the specified input and output layers.
	 */
	private double[][] randInitializeWeights(int in , int out) {
		
		double[][] result = new double[out][in + 1]; // +1 for the bias unit
		
		double epsilon = XAVIER_NORMALIZED_INIT / Math.sqrt(in + out);
		
		// Fill in each member of result with a random initialization
		for (int row = 0 ; row < result.length ; row++) {
			
			for (int column = 0 ; column < result[row].length ; column++) {
				
				result[row][column] = rand.nextDouble() * 2.0 * epsilon - epsilon;
				
			}
			
		}
		
		return result;
		
	}
}
