import java.util.ArrayList;

public class TestCase {

	private double[] input;
	private int expectedOutput;
	private String expectedOutputString;
	private String name; // Optional value, gives the test case a name
	
	private static int count = 0;
	private static ArrayList<String> allOutputs = new ArrayList<String>();
	
	public TestCase(double[] input , String expectedOutputString) {
		
		this.input = input;
		this.expectedOutputString = expectedOutputString;
		
		assignExpectedOutput();
		
	}
	
	public TestCase(double[] input , String expectedOutputString , String name) {
		
		this.input = input;
		this.expectedOutputString = expectedOutputString;
		this.name = name;
		
		assignExpectedOutput();
		
	}
	
	/**
	 * Set expectedOutput to next value in count if output DNE yet, if it does, assign it that value
	 */
	private void assignExpectedOutput() {
		
		if ((expectedOutput = allOutputs.indexOf(expectedOutputString)) == -1) {
			
			System.out.println("New possible output: " + expectedOutputString);
			expectedOutput = count++;
			allOutputs.add(expectedOutputString);
			
		}
		
	}
	
	public double[] getInput() {
		
		return input;
		
	}
	
	public int getExpectedOutput() {
		
		return expectedOutput;
		
	}
	
	public String getExpectedOutputString() {
		
		return expectedOutputString;
		
	}
	
	
	public static double[][] casesToX(TestCase[] cases) {
		
		int inputLength = cases[0].getInput().length;
		double[][] casesX = new double[cases.length][inputLength + 1];
		for (int i = 0 ; i < cases.length ; i++) {
			
			casesX[i][0] = 1; // bias column
			
			for (int j = 0 ; j < inputLength ; j++) {
				
				casesX[i][j + 1] = cases[i].getInput()[j];
				
			}
			
		}
		
		return casesX;
		
	}
	
	public static double[][] casesToY(TestCase[] cases) {
		
		int outputLength = allOutputs.size();
		double[][] casesY = new double[cases.length][outputLength];
		
		for (int i = 0 ; i < cases.length ; i++)
			casesY[i][cases[i].getExpectedOutput()] = 1;
		
		return casesY;
		
	}
	
}
