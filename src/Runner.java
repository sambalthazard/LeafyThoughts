
public class Runner {
	
	public static void main(String[] args) {
		
		final int IMAGE_WIDTH = 80;
		final int IMAGE_HEIGHT = 45;
		final int NUM_PIXEL_CHARACTERISTICS = 3; // R, G, and B
		final double TRAINING_RATIO = .8; // MODIFY FOR BUGFIXING
		final double VALIDATION_RATIO = 0;
		final double TEST_RATIO = .2; // MODIFY FOR BUGFIXING
		
		int[] leafIDBrainLayerSizes = {IMAGE_WIDTH * IMAGE_HEIGHT * NUM_PIXEL_CHARACTERISTICS , 35 , 20 , 3};
		
		//int[] testBrainLayerSizes = {5 , 4 , 5 , 3}; // FOR TESTING
		try {
			
			/*Brain testBrain = new SimpleBrain(testBrainLayerSizes , TRAINING_RATIO , VALIDATION_RATIO , TEST_RATIO); // FOR BUGFIXING
			String[] testInformation = {"5 2 8 3 2,a" , "8 2 9 0 9,b" , "1 9 8 2 2,c" , "2 9 1 8 9,a" , "3 8 2 8 0,c" , "2 9 1 3 9,b"}; // FOR BUGFIXING
			testBrain.loadData(testInformation); // FOR BUGFIXING
			testBrain.train(); // FOR BUGFIXING
			testBrain.test(); */ // FOR BUGFIXING
			
			ImageBrain leafIDBrain = new ImageBrain(leafIDBrainLayerSizes , 
					TRAINING_RATIO , VALIDATION_RATIO , TEST_RATIO);
			
			String[] leafIDInformation = {"images/leafData.txt" , "images/" , IMAGE_WIDTH + "" , IMAGE_HEIGHT + "" , "1"};
			leafIDBrain.loadData(leafIDInformation);
			
			leafIDBrain.train();
			
			leafIDBrain.test();
			
		} catch (BrainException e) {
			
			e.printStackTrace();
			
		}
		
	}
	
}
