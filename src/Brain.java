import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public abstract class Brain {
	
	private int numLabels;
	private int[] layerSizes;
	protected String[] dataFileLines;
	protected TestCase[] cases;
	protected TestCase[] trainingSet;
	protected TestCase[] validationSet;
	protected TestCase[] testSet;
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
		
		thetas = fmincg(thetas , x , y , layerSizes , lambda , MAX_TRAINING_ITERATIONS , RED);
		
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
		
		int count = 0; // Run length counter
		boolean lsFailed = false; // Whether a previous line search has failed
		// Left out: declaration of fX, as don't need this to-be-returned value currently
		double cost1 = costFunction(thetas , x , y , layerSizes , lambda);
		// Implied: static gradients = gradients calculated by that cost function execution
		SimpleMatrix gradientsUnrolled1 = unroll(gradients); // Unroll gradients for easier processing
		count += (length < 0 ? 1 : 0);
		SimpleMatrix s = gradientsUnrolled1.scale(-1); // Search direction
		double d1 = gradientsUnrolled1.transpose().mult(s).get(0 , 0); // Slope
		double z1 = red / (1 - d1); // Initial step
		
		double cost0;
		SimpleMatrix x0;
		SimpleMatrix gradientsUnrolled0;
		double cost2;
		SimpleMatrix gradientsUnrolled2;
		double d2;
		double z2;
		double cost3;
		double d3;
		double z3;
		int M;
		boolean success;
		double limit;
		
		while (count < length) {
			
			// Iter
			count += (length > 0 ? 1 : 0);
			
			// Copy current vals
			cost0 = cost1;
			x0 = x;
			gradientsUnrolled0 = gradientsUnrolled1;
			
			// Begin line search
			x = x.plus(s.scale(z1));
			
			cost2 = costFunction(thetas , x , y , layerSizes , lambda);
			gradientsUnrolled2 = unroll(gradients);
			
			count += (length < 0 ? 1 : 0);
			
			d2 = gradientsUnrolled2.transpose().mult(s).get(0 , 0);
			
			// Init point 3 equal to point 1
			cost3 = cost1;
			d3 = d1;
			z3 = z1 * -1;
			
			if (length > 0)
				M = MAX;
			else
				M = Math.min(MAX , -1 * length - count);
			
			success = false;
			limit = -1;
			
			while (true) {
				
				while (((cost2 > cost1 + z1 * RHO * d1) || (d2 > -1 * SIG * d1)) && M > 0) {
					
					limit = z1;
					
					if (cost2 > cost1) { // Then quadratic fit:
						
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + cost2 - cost3);
					
					}
					else { // Cubic fit:
						
						double a = 6 * (cost2 - cost3) / z3 + 3 * (d2 + d3);
						double b = 3 * (cost3 - cost2) - z3 * (d3 + 2 * d2);
						if (a != 0)
							z2 = (Math.sqrt(b * b - a * d2 * z3 * z3) - b) / a;
						else // In case of divide by 0
							z2 = z3 / 2;
						
					}
					
					z2 = Math.max(Math.min(z2 , INT * z3) , (1 - INT) * z3); // Don't accept too close to limits
					z1 += z2; // Update step
					
					// Continue line search
					x = x.plus(s.scale(z2));
					
					cost2 = costFunction(thetas , x , y , layerSizes , lambda);
					gradientsUnrolled2 = unroll(gradients);
					
					M -= 1;
					count += (length < 0 ? 1 : 0);
					
					d2 = gradientsUnrolled2.transpose().mult(s).get(0 , 0);
					z3 = z3 - z2; // z3 now relative to z2's location
					
				}
				
				if (cost2 > z1 * RHO * d1 || d2 > -1 * SIG * d1) // Failure
					break;
				else if (d2 > SIG * d1) { // Success
					
					success = true;
					break;
					
				}
				else if (M == 0) // Failure
					break;
				
				
				
			}
			
		}
		
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
	 * Feeds forward an input through NN layers to produce NN's answers.
	 * @param thetas The weights for each forward feed through NN layers.
	 * @param a In index 0, the matrix to feed forward through the NN.
	 * @return The calculated answers for each case (% probability in each index) in each layer. 
	 */
	private static SimpleMatrix[] feedForward(SimpleMatrix[] thetas , SimpleMatrix[] a) {
		
		for (int i = 0 ; i < thetas.length ; i++) {
			
			// Feedforward one layer
			z[i + 1] = a[i].mult(thetas[i].transpose());
			a[i + 1] = sigmoid(z[i + 1]);
			
			// If not the last feedforward step, then append a bias column of 1s
			if (i + 1 < thetas.length) {
				
				// Workaround to SimpleMatrix limitation of no left-growing:
				//	(0 , 0)		(1 , 0)		(1  , a00, .. , a0j)
				//	(0 , 0)	->	(1 , 0)	->	(1  , a10, .. , a1j)
				//	(. , .)		(. , .)		(.  , .  , .. , .  )
				//	(0 , 0)		(1 , 0)		(1  , ai0, .. , aij)
				a[i + 1] = (new SimpleMatrix(a[i + 1].numRows() , 2))
						.combine(0 , 0 , new SimpleMatrix(a[i + 1].numRows() , 1).plus(1))
						.combine(0 , 1 , a[i]);
				
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
			
			SimpleMatrix sigGrad = sigmoidGradient(z[i]);
			ds[i - 1] = thetas[i].transpose().mult(ds[i]);
			ds[i - 1] = ds[i - 1].extractMatrix(1 , SimpleMatrix.END , 0 , SimpleMatrix.END).elementMult(sigGrad);
			
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
	 */
	private static SimpleMatrix unroll(SimpleMatrix[] mxs) {
		
		SimpleMatrix result = new SimpleMatrix(0 , 1);
		for (SimpleMatrix mx : mxs) {
			
			result = result.combine(SimpleMatrix.END , 1 , unroll(mx));
			
		}
		
		return result;
		
	}
	
	/**
	 * 
	 * @param mx
	 */
	private static SimpleMatrix unroll(SimpleMatrix mx) {
		
	    mx.reshape(1, mx.getNumElements());
	    return mx;
		
	}
	
	/**
	 * Calculates the sigmoid equation for each element in a SimpleMatrix and returns the result.
	 * @param mx The matrix to apply the sigmoid equation to each element.
	 * @return The input matrix with sigmoid equation applied to each element
	 */
	public static SimpleMatrix sigmoid(SimpleMatrix mx) {
		
		// Equivalent to 1 / (1 + e^(mx)) for each element in mx
		return mx.scale(-1).elementExp().plus(1).elementPower(-1);
		
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
		
		// Fill in each member of result with a random initialization
		for (int row = 0 ; row < result.length ; row++) {
			
			for (int column = 0 ; column < result[row].length ; column++) {
				
				result[row][column] = rand.nextDouble() * 2.0 * epsilon - epsilon;
				
			}
			
		}
		
		return new SimpleMatrix(result);
		
	}
}
