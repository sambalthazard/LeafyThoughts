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
		
		// Set expectedOutput to next value in count if output DNE yet, if it does, assign it that value
		if ((expectedOutput = allOutputs.indexOf(expectedOutputString)) == -1) {
			
			expectedOutput = count++;
			allOutputs.add(expectedOutputString);
			
		}
		
	}
	
	public TestCase(double[] input , String expectedOutputString , String name) {
		
		this.input = input;
		this.expectedOutputString = expectedOutputString;
		this.name = name;
		
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
	
}
