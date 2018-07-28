import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.ejml.data.*;

public abstract class Brain {
	
	private int numLabels;
	private int[] layerSizes;
	protected String[] dataFileLines;
	protected TestCase[] cases;
	protected TestCase[] trainingCases;
	protected TestCase[] validationCases;
	protected TestCase[] testCases;
	private double trainingSplit; // The ratio of input cases for the training set
	private double validationSplit; // "" validation set
	private double testSplit; // "" test set
	protected String[] labels; // All the possible output values of the brain
	private SimpleMatrix[] thetas;
	private Random rand; // Random number generator
	private double lambda; // Weight regularization parameter
	
	private static SimpleMatrix[] gradients; // Most recent gradients calculated by costFunction
	private static SimpleMatrix[] z; // Most recent z values calculated by feedForward; z[i] = a[i] before sigmoid

	public static final double XAVIER_NORMALIZED_INIT = Math.sqrt(6);
	
	// fmingc constants:
	public static final int MAX_TRAINING_ITERATIONS = 50;
	public static final int RED = 1;
	private static final double RHO = 0.01;
	private static final double SIG = 0.5; // RHO and SIG are the constants in the Wolfe-Powell conditions
	private static final double INT = 0.1; // Don't reevaluate within 0.1 of the limit of the current bracket
	private static final double EXT = 3.0; // Extrapolate maximum 3 times the current bracket
	private static final int MAX = 20; // Max 20 function evaluations per line search
	private static final double RATIO = 100; // Maximum allowed slope ratio
	
	/**
	 * Initializes the brain.
	 * @param layerSizes The number of nodes in each layer of the network. Index 0 is the input layer, and the last index is the output layer.
	 * @param trainingRatio 
	 * @param validationRatio 
	 * @param testRatio 
	 * @throws BrainException
	 */
	public Brain(int[] layerSizes , double trainingRatio , double validationRatio , double testRatio) throws BrainException {
		
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
			
			// Initialize training/validation/test sets and ratios
			trainingCases = new TestCase[0];
			validationCases = new TestCase[0];
			testCases = new TestCase[0];
			trainingSplit = trainingRatio;
			validationSplit = validationRatio;
			testSplit = testRatio;
			
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
	 * @throws BrainException 
	 */
	public void train() throws BrainException {
		
		splitCases(); // Split cases into training, validation, and test sets
		
		if (trainingCases.length < 1)
			throw new BrainException("Can't train the brain with 0 training cases! Try increasing the trainingRatio.");
		
		// Set case inputs as matrix of inputs with a bias column of 1s
		double[][] trainingX = TestCase.casesToX(trainingCases);
		double[][] validationX = TestCase.casesToX(validationCases);
		
		// Recode known answers as an array of vectors with 1 at the index of the y value, 0 at all other indices
		double[][] trainingY = TestCase.casesToY(trainingCases);
		double[][] validationY = TestCase.casesToY(validationCases);
		
		// Convert into SimpleMatrix objects for efficient matrix computation
		SimpleMatrix trainingMxX = new SimpleMatrix(trainingX);
		SimpleMatrix trainingMxY = new SimpleMatrix(trainingY);
		/*SimpleMatrix validationMxX = new SimpleMatrix(validationX); // STILL NEED TO USE THESE TO TELL FMINCG WHEN TO STOP
		SimpleMatrix validationMxY = new SimpleMatrix(validationY);*/
		
		// Train from training cases
		thetas = fmincg(thetas , trainingMxX , trainingMxY , layerSizes , lambda , MAX_TRAINING_ITERATIONS , RED);
		
		// **** USE VALIDATION SET **** //
		
	}
	
	/**
	 * Tests the accuracy of the trained NN by feeding forward the test set through the NN.
	 * @throws BrainException 
	 */
	public void test() throws BrainException {
		
		if (testCases.length < 1)
			throw new BrainException("Can't test the brain with 0 test cases! Try increasing the testRatio.");
		
		// Set test case inputs as matrix of inputs with a bias column of 1s
		double[][] testX = TestCase.casesToX(testCases);
		
		// Recode known test answers as an array of vectors with 1 at the index of the y value, 0 at all other indices
		// double[][] testY = TestCase.casesToY(testCases);
		
		// Convert into SimpleMatrix objects for efficient matrix computation
		SimpleMatrix testMxX = new SimpleMatrix(testX);
		
		// Feed forward to find NN's predicted outputs
		SimpleMatrix output = feedForwardSimple(thetas , testMxX); // Transpose to make it 3x12
		int[] dimensions = {output.numRows() , output.numCols()};
		double[][] outputDouble = reshapeToDouble(output , dimensions , false)[0];
		
		// Compare with expected outputs
		int match = 0;
		int total = 0;
		for (int i = 0 ; i < testCases.length ; i++) {
			
			if (getIndexOfMax(outputDouble[i]) == testCases[i].getExpectedOutput())
				match++;
			
			total++;
			
		}
		
		double accuracy = ((double) match) / ((double) total);
		
		System.out.println(accuracy);
		
	}
	
	/**
	 * Calculates the NN error, i.e. the cost J. Calculates theta gradients to modify weights in the right direction.
	 * @param thetas The weights for each forward feed through NN layers.
	 * @param x
	 * @param y The correct answers for each case (0 in index of incorrect answer, 1 in index of correct answer).
	 * @param layerSizes The number of nodes in each layer of the network. Index 0 is the input layer, and the last index is the output layer.
	 * @param lambda The regularization parameter.
	 * @return The cost J; sets the static class variable 'gradients' to the gradients calculated.
	 */
	private static double costFunction(SimpleMatrix[] thetas , SimpleMatrix x , SimpleMatrix y , int[] layerSizes , double lambda) {
		
		int m = x.numRows();
		int numLabels = layerSizes[layerSizes.length - 1];
		
		// Copy x into a for modification in feedforward
		SimpleMatrix[] a = new SimpleMatrix[layerSizes.length];
		a[0] = new SimpleMatrix(x);
		
		// Initialize z
		z = new SimpleMatrix[thetas.length];
		
		// Feedforward through the layers:
		a = feedForward(thetas , a);
		
		// Calculate non-regularized cost j
		double j = calculateCost(a[a.length - 1] , y , m);
		
		// Regularize cost j
		j = regularizeCost(j , thetas , m , lambda);
		
		// Backpropagate to get non-regularized gradients
		gradients = backpropagate(a , x , y , thetas , m);
		
		// Regularize gradients
		regularizeGradients(thetas , lambda , m);
		
		return j;
		
	}
	
	/**
	 * Feeds forward an input through NN layers to produce NN's answers and all intermediary values.
	 * @param thetas The weights for each forward feed through NN layers.
	 * @param a In index 0, the matrix to feed forward through the NN.
	 * @return The calculated answers for each case (% probability in each index) in each layer. 
	 */
	private static SimpleMatrix[] feedForward(SimpleMatrix[] thetas , SimpleMatrix[] a) {
		
		for (int i = 0 ; i < thetas.length ; i++) {
			
			// Feedforward one layer
			z[i] = a[i].mult(thetas[i].transpose());
			SimpleMatrix zcopy = z[i];
			a[i + 1] = sigmoid(z[i]);
			
			// If not the last feedforward step, then append a bias column of 1s
			if (i + 1 < thetas.length) {
				
				// Workaround to SimpleMatrix limitation of no left-growing:
				//	(0 , 0)		(1 , 0)		(1  , a00, .. , a0j)
				//	(0 , 0)	->	(1 , 0)	->	(1  , a10, .. , a1j)
				//	(. , .)		(. , .)		(.  , .  , .. , .  )
				//	(0 , 0)		(1 , 0)		(1  , ai0, .. , aij)
				a[i + 1] = (new SimpleMatrix(a[i + 1].numRows() , 2))
						.combine(0 , 0 , new SimpleMatrix(a[i + 1].numRows() , 1).plus(1))
						.combine(0 , 1 , a[i + 1]);
				
			}
			
		}
		
		return a;
		
	}
	
	/**
	 * Feeds forward an input through NN layers to produce NN's answers.
	 * @param thetas The weights for each forward feed through NN layers.
	 * @param a In index 0, the matrix to feed forward through the NN.
	 * @return The calculated answers for each case (% probability in each index) in JUST the output layer. 
	 */
	private static SimpleMatrix feedForwardSimple(SimpleMatrix[] thetas , SimpleMatrix a) {
		
		for (int i = 0 ; i < thetas.length ; i++) {
			
			// Feedforward one layer (no assignment to z)
			a = sigmoid(a.mult(thetas[i].transpose()));
			
			// If not the last feedforward step, then append a bias column of 1s
			if (i + 1 < thetas.length) {
				
				a = (new SimpleMatrix(a.numRows() , 2))
						.combine(0 , 0 , new SimpleMatrix(a.numRows() , 1).plus(1))
						.combine(0 , 1 , a);
				
			}
			
		}
		
		return a;
		
	}
	
	/**
	 * Calculates the cost J of a's distance from y, i.e. how far off the feedforward algorithm is from 100% accurate.
	 * @param a The calculated answers for each case (% probability in each index).
	 * @param y The correct answers for each case (0 in index of incorrect answer, 1 in index of correct answer).
	 * @param m The number of cases.
	 * @return The cost J of a's distance from y.
	 */
	private static double calculateCost(SimpleMatrix a , SimpleMatrix y , int m ) {
		
		int numLabels = y.numCols();
		
		// Compute cost: J = (1/m) * sum(1->m)sum(1->k) [ -yki log((hθ(xi))k) - (1 - yki) log(1 - (hθ(xi))k)]
		int j = 0;
		for (int i = 0 ; i < numLabels ; i++) {
		
			SimpleMatrix minusYiVertical = y.extractVector(false , i).scale(-1);
			SimpleMatrix hOfXiHorizontal = a.extractVector(false , i).transpose();
			
			j += (hOfXiHorizontal.elementLog().mult(minusYiVertical)
					.minus(hOfXiHorizontal.scale(-1).plus(1).elementLog().mult(minusYiVertical.plus(1))))
				.elementSum();
			
		}
		j *= (1.0 / m);
		
		return j;
		
	}
	
	/**
	 * Computes cost J with regularization of degree 2 from non-regularized J.
	 * @param jNonReg Initial, non-regularized J.
	 * @param thetas The weights for each forward feed through NN layers.
	 * @param m The number of cases.
	 * @param lambda The regularization parameter.
	 * @return The regularized cost J.
	 */
	private static double regularizeCost(double jNonReg , SimpleMatrix[] thetas , int m , double lambda) {
		
		// Regularize: J = jNonReg + (lambda / 2m) * [each element of thetas ^2]
		double thetasSquaredSum = 0;
		for (SimpleMatrix mx : thetas)
			thetasSquaredSum += mx.elementPower(2).elementSum();
		double jReg = jNonReg + (lambda / (2.0 * m)) * thetasSquaredSum;
		
		return jReg;
		
	}
	
	/**
	 * 
	 * @param a
	 * @param x
	 * @param y
	 * @param thetas
	 * @param m
	 * @return
	 */
	private static SimpleMatrix[] backpropagate(SimpleMatrix[] a , SimpleMatrix x , SimpleMatrix y , SimpleMatrix[] thetas , int m) {
		
		int numThetas = thetas.length;
		SimpleMatrix[] deltas = new SimpleMatrix[numThetas]; // Big D matrices
		SimpleMatrix[] ds = new SimpleMatrix[numThetas]; // Little d matrices (1 vector per case)
		
		ds[numThetas - 1] = a[numThetas].minus(y).transpose();
		
		// Calculate little d's
		for (int i = numThetas - 1 ; i > 0 ; i--) {
			
			SimpleMatrix sigGrad = sigmoidGradient(z[i - 1]);
			ds[i - 1] = thetas[i].transpose().mult(ds[i]);
			ds[i - 1] = ds[i - 1].extractMatrix(1 , SimpleMatrix.END , 0 , SimpleMatrix.END).elementMult(sigGrad.transpose());
			
		}
		
		// Calculate big D's and then the resulting (unregularized) gradient
		for (int i = 0 ; i < deltas.length ; i++) {
			
			deltas[i] = ds[i].mult(a[i]).divide(m);
			
		}
		
		return deltas;
		
	}
	
	/**
	 * 
	 * @param thetas
	 * @param lambda
	 * @param m
	 * @return
	 */
	private static SimpleMatrix[] regularizeGradients(SimpleMatrix[] thetas , double lambda , int m) {
		
		for (int i = 0 ; i < gradients.length ; i++) {
			
			SimpleMatrix biasColumn = gradients[i].extractVector(false , 0);
			gradients[i] = gradients[i].scale(1.0 + (lambda / m)); // Regularization
			gradients[i] = gradients[i].combine(0 , 0 , biasColumn); // Don't regularize the bias column
			
		}
		
		return gradients;
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	private static SimpleMatrix unroll(SimpleMatrix[] mxs) {
		
		// TESTED: Conversion to double -> processing -> reconversion to SimpleMatrix takes ~0.36 as long as SimpleMatrix's reshape method
		int unrolledLength = 0;
		
		// Determine total elements in matrices
		for (SimpleMatrix mx : mxs)
			unrolledLength += mx.numCols() * mx.numRows();
		
		// Create 2D array for unrolled matrix storage; will only use 1D, but need 2D to convert back to SimpleMatrix
		double[][] doubleResult = new double[1][unrolledLength];
		
		// Unroll each matrix and append it to doubleResult array
		int index = 0;
		for (SimpleMatrix mx : mxs) {
			
			double[] mxArray = unrollToDouble(mx);
			System.arraycopy(mxArray , 0 , doubleResult[0] , index , mxArray.length); // Copy in 
			index += mxArray.length;
			
		}
		
		// Reconvert double array to SimpleMatrix, transpose it so its dimensions are (length x 1)
		SimpleMatrix result = (new SimpleMatrix(doubleResult)).transpose();
		
		return result;
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	private static SimpleMatrix unroll(SimpleMatrix mx) {
		
		int unrolledLength = mx.numCols() * mx.numRows();
	    double[][] doubleResult = new double[1][unrolledLength];
	    doubleResult[0] = unrollToDouble(mx);
	    return (new SimpleMatrix(doubleResult)).transpose();
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	private static double[] unrollToDouble(SimpleMatrix mx) {
		
		DMatrixD1 d1Mx = mx.getMatrix();
		return d1Mx.data;
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	private static SimpleMatrix[] reshape(SimpleMatrix mx , int[] dimensions , boolean vertical) {
		
		SimpleMatrix[] reshaped = new SimpleMatrix[dimensions.length / 2];
		
		// Unroll matrix to "raw" double array for fast double[] processing
		double[] raw = unrollToDouble(mx);
		
		int start = 0;
		for (int i = 0 ; i < reshaped.length ; i++) {
			
			int cols = dimensions[i * 2];
			int rows = dimensions[i * 2 + 1];
			
			// If unrolling horizontally, switch cols and rows for horizontal processing
			if (!vertical) {
				
				int temp = cols;
				cols = rows;
				rows = temp;
				
			}
			
			double[][] reshapedDouble = new double[cols][rows];
			for (int j = 0 ; j < cols ; j++) {
				
				// Copy one column from raw->reshapedDouble[i]
				System.arraycopy(raw , start , reshapedDouble[j] , 0 , rows);
				
				// Update start
				start += rows;
				
			}
			
			// Build a SimpleMatrix from reshaped double[][]
			reshaped[i] = new SimpleMatrix(reshapedDouble).transpose();
			
		}
		
		// If unrolling horizontally, transpose back to the right dimensions before return
		if (!vertical) {
			
			for (int i = 0 ; i < reshaped.length ; i++)
				reshaped[i] = reshaped[i].transpose();
			
		}
		
		return reshaped;
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	private static double[][][] reshapeToDouble(SimpleMatrix mx , int[] dimensions , boolean vertical) {
		
		double[][][] reshaped = new double[dimensions.length / 2][][];
		
		// Unroll matrix to "raw" double array
		double[] raw = unrollToDouble(mx);
		
		int start = 0;
		for (int i = 0 ; i < reshaped.length ; i++) {
			
			int cols = dimensions[i * 2];
			int rows = dimensions[i * 2 + 1];
			
			// If unrolling horizontally, switch cols and rows for processing
			if (!vertical) {
				
				int temp = cols;
				cols = rows;
				rows = temp;
				
			} // RETHINK: has to do more than this
			
			reshaped[i] = new double[cols][rows];
			for (int j = 0 ; j < cols ; j++) {
				
				// Copy one column from raw->reshapedDouble[i]
				System.arraycopy(raw , start , reshaped[i][j] , 0 , rows);
				
				// Update start
				start += rows;
				
			}
			
		}
		// If unrolling horizontally, transpose back to the right dimensions before return
		if (!vertical) {
			
			// Save pre-transposed matrices
			double[][][] temp = reshaped;
			// Overwrite output matrices so they can be reshaped to transposed dimensions
			reshaped = new double[reshaped.length][][];
			
			// For each matrix
			for (int i = 0 ; i < reshaped.length ; i++) {
				
				// Create empty matrix w/ transposed dimensions
				double[][] current = new double[temp[i][0].length][temp[i].length];
				
				for (int j = 0 ; j < temp[i][0].length ; j++) {
					
					for (int k = 0 ; k < temp[i].length ; k++) {
						
						// Copy element from pre-transposed matrix into its transposed position
						current[j][k] = temp[i][k][j];
						
					}
					
				}
				
				reshaped[i] = current;
				
			}
			
		}
				
		return reshaped;
		
	}
	
	/**
	 * 
	 * @param toScale
	 * @return
	 * @throws BrainException
	 */
	public static double[] featureScale(double[] toScale) throws BrainException {
		
		double mean = getMean(toScale);
		double stdev = getStDev(toScale);
		
		for (int i = 0 ; i < toScale.length ; i++)
			toScale[i] = (toScale[i] - mean) / stdev;
		
		return toScale;
		
	}
	
	/**
	 * 
	 * @param array
	 * @return
	 */
	public static int getIndexOfMax(double[] array) {
		
		int index = 0;
		double max = array[0];
		
		for (int i = 1 ; i < array.length ; i++) {
			
			if (array[i] > max) {
				
				index = i;
				max = array[i];
				
			}
			
		}
		
		return index;
		
	}
	
	/**
	 * 
	 * @param array
	 * @return
	 * @throws BrainException
	 */
	public static double getMean(double[] array) throws BrainException {
		
		double n = array.length;
		
		if (n <= 0)
			throw new BrainException("Array contents empty, no standard deviation.");
		
		double total = 0;
		for (int i = 0 ; i < array.length ; i++)
			total += array[i];
		
		return total / n;
		
	}
	
	/**
	 * Calculates the standard deviation of a matrix of doubles.
	 * @param matrix Matrix of doubles to calculate standard deviation of.
	 * @return
	 * @throws BrainException 
	 * @precondition matrix rows all same length, and columns all same length.
	 */
	public static double getStDev(double[] array) throws BrainException {

        double n = array.length;
        
		if (n <= 0)
			throw new BrainException("Array contents empty, no standard deviation.");
		
		double mean = getMean(array);
        double total = 0;

        for (int i = 0 ; i < array.length ; i++ )
                total += Math.pow((array[i] - mean) , 2);
        
        return Math.sqrt(total / (n - 1.0));
		
	}
	
	/**
	 * Calculates the sigmoid equation for each element in a SimpleMatrix and returns the result.
	 * @param mx The matrix to apply the sigmoid equation to each element.
	 * @return The input matrix with sigmoid equation applied to each element
	 */
	public static SimpleMatrix sigmoid(SimpleMatrix mx) {
		
		// Equivalent to 1 / (1 + e^(mx)) for each element in mx
		SimpleMatrix one = mx.scale(-1);
		SimpleMatrix two = one.elementExp();
		SimpleMatrix three = two.plus(1);
		SimpleMatrix four = three.elementPower(-1.0);
		return mx.scale(-1).elementExp().plus(1).elementPower(-1.0);
		
	}
	
	/**
	 * 
	 * @param mx
	 * @return
	 */
	public static SimpleMatrix sigmoidGradient(SimpleMatrix mx) {
		
		SimpleMatrix sig = sigmoid(mx);
		return sig.elementMult(sig.scale(-1).plus(1));
		
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
			int actualEnd;
			for (actualEnd = content.length ; content[actualEnd - 1] == null && actualEnd > 0 ; actualEnd--);
			content = Arrays.copyOfRange(content , 0 , actualEnd);
			
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
	 * Randomly splits all the cases provided into training set, validation set, and test set with instance variable ratios
	 * @throws BrainException 
	 */
	protected void splitCases() throws BrainException {
		
		// First, determine the start and end points of the cases set for each subset
		int numCases = cases.length;
		int trainingEnd = (int) (numCases * trainingSplit);
		int validationEnd = (int) (trainingEnd + numCases * validationSplit);
		
		// Before allocating the cases to the subsets, shuffle the cases:
		shuffleArray(cases);
		
		// Then, allocate the appropriate number of cases to each subset:
		if (trainingEnd > 0)
			trainingCases = Arrays.copyOfRange(cases , 0 , trainingEnd);
		
		if (validationEnd > trainingEnd)
			validationCases = Arrays.copyOfRange(cases , trainingEnd , validationEnd);
		
		if (cases.length > validationEnd)
			testCases = Arrays.copyOfRange(cases , validationEnd , cases.length);
		
	}
	
	/**
	 * Shuffles an array of any type using the Fisher-Yates algorithm.
	 * @param array
	 * @return
	 */
	public static <T> T[] shuffleArray(T[] array) {
		
		Random r = new Random();
		
		for (int i = array.length - 1 ; i > 0 ; i--) {
			
			int index = r.nextInt(i + 1);
			
			// Swap
			T temp = array[index];
			array[index] = array[i];
			array[i] = temp;
			
		}
		
		return array;
		
	}
	
	/**
	 * Creates a matrix of randomly initialized weights for every NN connection specified by the layer sizes.
	 * @return The 3D array of randomly initialized weights.
	 */
	private SimpleMatrix[] randInitializeNNWeights() {
		
		SimpleMatrix[] result = new SimpleMatrix[layerSizes.length - 1];
		for (int x = 0 ; x < layerSizes.length - 1 ; x++) {
			
			result[x] = randInitializeWeights(layerSizes[x] , layerSizes[x + 1]);
			
		}
		
		return result;
		
	}
	
	/**
	 * Creates a matrix of randomly initialized weights (theta) for the connections between layers of specified sizes, in->out.
	 * @param in The size of the input layer of the connection.
	 * @param out The size of the output layer of the connection.
	 * @return The matrix of randomly initialized weights (theta) for the connections between input and output layers of the specified sizes.
	 */
	private SimpleMatrix randInitializeWeights(int in , int out) {
		
		double[][] result = new double[out][in + 1]; // +1 for the bias unit
		
		double epsilon = XAVIER_NORMALIZED_INIT / Math.sqrt(in + out);
		
		double val = 0; // TEMP, FOR BUGFIXING
		
		// Fill in each member of result with a random initialization
		for (int row = 0 ; row < result.length ; row++) {
			
			for (int column = 0 ; column < result[row].length ; column++) {
				
				//result[row][column] = rand.nextDouble() * 2.0 * epsilon - epsilon;
				result[row][column] = (val - (int) val) * 2.0 * epsilon - epsilon; // TEMP, FOR BUGFIXING
				val+= .1; // TEMP, FOR BUGFIXING
				
			}
			
		}
		
		return new SimpleMatrix(result);
		
	}


	/**
	 * 
	 * NOT MY ALGORITHM. I TRANSLATED THIS FROM MATLAB USING MY RELEVANT DATA TYPES.
	 * Copyright (C) 1999, 2000, & 2001, Carl Edward Rasmussen
	 * @param thetas
	 * @param x
	 * @param y
	 * @param layerSizes
	 * @param lambda
	 * @param maxIterations
	 * @param red
	 * @return
	 */
	private static SimpleMatrix[] fmincg(SimpleMatrix[] thetas , SimpleMatrix x , SimpleMatrix y , int[] layerSizes , double lambda , int maxIterations , int red) {
		
		int length = maxIterations;
		
		SimpleMatrix weights = unroll(thetas);
		// Set dimensions for reshape
		int[] dimensions = new int[thetas.length * 2];
		for (int i =  0 ; i < thetas.length ; i++) {
		
			dimensions[i * 2] = thetas[i].numCols();
			dimensions[i * 2 + 1] = thetas[i].numRows();
				
		}
		
		int count = 0; // Run length counter
		boolean lsFailed = false; // Whether a previous line search has failed
		// Left out: declaration of fX, as don't need this to-be-returned value currently
		double cost1 = costFunction(reshape(weights , dimensions , false) , x , y , layerSizes , lambda);
		// Implied: static gradients = gradients calculated by that cost function execution
		SimpleMatrix gradientsUnrolled1 = unroll(gradients); // Unroll gradients for easier processing
		count += (length < 0 ? 1 : 0);
		SimpleMatrix s = gradientsUnrolled1.scale(-1); // Search direction
		double slope1 = gradientsUnrolled1.transpose().mult(s).get(0 , 0); // Slope
		double z1 = red / (1 - slope1); // Initial step
		
		double cost0;
		SimpleMatrix weights0;
		SimpleMatrix gradientsUnrolled0;
		double cost2;
		SimpleMatrix gradientsUnrolled2;
		double slope2;
		double z2;
		double cost3;
		double slope3;
		double z3;
		int M;
		boolean success;
		double limit;
		double a;
		double b;
		
		while (count < Math.abs(length)) { // *****TEST WITH ALL WEIGHTS INIT AT 0 BOTH HERE AND MATLAB, compare step by step*****
			
			// Iter
			count += (length > 0 ? 1 : 0);
			
			// Copy current vals
			cost0 = cost1;
			weights0 = weights;
			gradientsUnrolled0 = gradientsUnrolled1;
			
			// Begin line search
			weights = weights.plus(s.scale(z1));
			
			cost2 = costFunction(reshape(weights , dimensions , false) , x , y , layerSizes , lambda);
			gradientsUnrolled2 = unroll(gradients);
			
			count += (length < 0 ? 1 : 0);
			
			slope2 = gradientsUnrolled2.transpose().mult(s).get(0 , 0);
			
			// Init point 3 equal to point 1
			cost3 = cost1;
			slope3 = slope1;
			z3 = z1 * -1.0;
			
			if (length > 0)
				M = MAX;
			else
				M = Math.min(MAX , -1 * length - count);
			
			success = false;
			limit = -1;
			
			while (true) {
				
				while (((cost2 > cost1 + z1 * RHO * slope1) || (slope2 > -1 * SIG * slope1)) && M > 0) {
					
					limit = z1;
					
					if (cost2 > cost1) { // Then quadratic fit:
						
						z2 = z3 - (0.5 * slope3 * z3 * z3) / (slope3 * z3 + cost2 - cost3);
					
					}
					else { // Cubic fit:
						
						a = 6.0 * (cost2 - cost3) / z3 + 3 * (slope2 + slope3);
						b = 3.0 * (cost3 - cost2) - z3 * (slope3 + 2 * slope2);
						if (a != 0)
							z2 = (Math.sqrt(b * b - a * slope2 * z3 * z3) - b) / a;
						else // In case of divide by 0
							z2 = z3 / 2;
						
					}
					
					z2 = Math.max(Math.min(z2 , INT * z3) , (1.0 - INT) * z3); // Don't accept too close to limits
					z1 += z2; // Update step
					
					// Continue line search
					weights = weights.plus(s.scale(z2));
					
					cost2 = costFunction(reshape(weights , dimensions , false) , x , y , layerSizes , lambda);
					gradientsUnrolled2 = unroll(gradients);
					
					M -= 1.0;
					count += (length < 0 ? 1 : 0);
					
					slope2 = gradientsUnrolled2.transpose().mult(s).get(0 , 0);
					z3 = z3 - z2; // z3 now relative to z2's location
					
				}
				
				if (cost2 > cost1 + z1 * RHO * slope1 || slope2 > -1.0 * SIG * slope1) // Failure
					break;
				else if (slope2 > SIG * slope1) { // Success
					
					success = true;
					break;
					
				}
				else if (M == 0) // Failure
					break;
				
				// Cubic extrapolation
				a = 6.0 * (cost2 - cost3) / z3 + 3 * (slope2 + slope3);
				b = 3.0 * (cost3 - cost2) - z3 * (slope3 + 2 * slope2);
				
				double sqrtEval = b * b - a * slope2 * z3 * z3;
				double denom; // Only evaluate once decided sqrt will be real
				if (sqrtEval < 0 || (denom = b + Math.sqrt(sqrtEval)) == 0) { // If sqrt imaginary or div by 0
					
					if (limit < -0.5) // If no upper limit
						z2 = z1 * (EXT - 1.0); // Extrapolate max amount
					else
						z2 = (limit - z1) / 2.0; // Otherwise bisect
					
				} else {
					
					z2 = -1.0 * slope2 * z3 * z3 / denom; // Intended calculation, no errors
					
					if (limit > -0.5 && z2 + z1 > limit) // If extrapolation beyond max
						z2 = (limit - z1) / 2.0; // Then bisect
					else if (limit < -0.5 && z2 + z1 > z1 * EXT) // If extrapolation beyond limit
						z2 = z1 * (EXT - 1.0); // Set to extrapolation limit
					else if (z2 < -1.0 * z3 * INT)
						z2 = -1.0 * z3 * INT;
					else if (limit > -0.5 && z2 < (limit - z1) * (1.0 - INT)) // If too close to limit
						z2 = (limit - z1) * (1.0 - INT);
					
				}
				
				cost3 = cost2; // Set point 3 <- point 2
				slope3 = slope2;
				z3 = -z2;
				
				z1 = z1 + z2; // Update current estimates
				weights = weights.plus(s.scale(z2));
				
				cost2 = costFunction(reshape(weights , dimensions , false) , x , y , layerSizes , lambda);
				gradientsUnrolled2 = unroll(gradients);
				
				M = M - 1;
				count += (length < 0 ? 1 : 0);
				
				slope2 = gradientsUnrolled2.transpose().mult(s).get(0 , 0);
				
			} // END of line search
			
			if (success) {
				
				cost1 = cost2;
				System.out.println("Iter cost: " + cost1);
				
				// Polack-Ribiere direction:
				s = s.scale(
						(gradientsUnrolled2.transpose().mult(gradientsUnrolled2).get(0 , 0)
						- gradientsUnrolled1.transpose().mult(gradientsUnrolled2).get(0 , 0))
						/ (gradientsUnrolled1.transpose().mult(gradientsUnrolled1).get(0 , 0))
						).minus(gradientsUnrolled2);
				
				// Swap derivatives:
				SimpleMatrix temp = gradientsUnrolled1.copy();
				gradientsUnrolled1 = gradientsUnrolled2;
				gradientsUnrolled2 = temp;
				
				slope2 = gradientsUnrolled1.transpose().mult(s).get(0 , 0); // Calculate new slope
				if (slope2 > 0) { // If new slope is not negative (it must be), use steepest direction:
					
					s = gradientsUnrolled1.scale(-1);
					slope2 = s.transpose().scale(-1).mult(s).get(0 , 0);
					
				}
				
				z1 *= Math.min(RATIO , slope1 / (slope2 - Double.MIN_VALUE)); // Slope ratio, capped at RATIO
				slope1 = slope2;
				lsFailed = false;
				
			} else { // Failure
				
				weights = weights0; // Set point 1 <- before failed line search
				cost1 = cost0;
				gradientsUnrolled1 = gradientsUnrolled0;
				
				if (lsFailed || count > Math.abs(length)) // If ls failed twice in a row or run out of time:
					break;
				
				// Swap derivatives:
				SimpleMatrix temp = gradientsUnrolled1.copy();
				gradientsUnrolled1 = gradientsUnrolled2;
				gradientsUnrolled2 = temp;
				
				s = gradientsUnrolled1.scale(-1); // Try steepest direction
				slope2 = s.transpose().scale(-1).mult(s).get(0 , 0);
				
				z1 = 1.0 / (1.0 - slope1);
				
				lsFailed = true;
				
			}
			
		}
		
		return reshape(weights , dimensions , false);
		
	} // END fmincg
	
} // END class Brain