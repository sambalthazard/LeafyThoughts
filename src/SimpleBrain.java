

public class SimpleBrain extends Brain {
	
	private final int PRE_DATA_LINES = 1;
	
	private int[][] rgb; // Defined in loadData, NOT in constructor
	private int[] unrolledRgb; // Defined in loadData, NOT in constructor

	/**
	 * 
	 * @param layerSizes The number of nodes in each layer of the network. Index 0 is the input layer, and the last number is the output layer.
	 * @param trainingRatio 
	 * @param validationRatio 
	 * @param testRatio 
	 * @throws BrainException
	 */
	public SimpleBrain(int[] layerSizes , double trainingRatio , double validationRatio , double testRatio) throws BrainException {

		super(layerSizes , trainingRatio , validationRatio , testRatio);
		
	}
	
	/**
	 * Loads training data from default string form into test case objects.
	 * @param information Training data in format: one case per index, each index of form: "input1 input2 ... inputk,expectedOutput" where all inputs are real numbers.
	 */
	public void loadData(String[] information) throws BrainException {
		
		super.cases = new TestCase[information.length];
		
		for (int i = 0 ; i < information.length ; i++) {
			
			String[] inputsOutput = information[i].split(",");
			String[] inputsStrings = inputsOutput[0].split(" ");
			double[] inputs = new double[inputsStrings.length];
			for (int j = 0 ; j < inputsStrings.length ; j++)
				inputs[j] = Double.parseDouble(inputsStrings[j]);
			
			super.cases[i] = new TestCase(inputs , inputsOutput[1]);
			
		}
		
	}
	
}