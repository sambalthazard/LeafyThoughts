
public class Runner {
	
	public static void main(String[] args) {
		
		final int IMAGE_WIDTH = 80;
		final int IMAGE_HEIGHT = 45;
		final int NUM_PIXEL_CHARACTERISTICS = 3; // R, G, and B
		final double TRAINING_RATIO = .8;
		final double VALIDATION_RATIO = 0;
		final double TEST_RATIO = .2;
		
		int[] leafIDBrainLayerSizes = {IMAGE_WIDTH * IMAGE_HEIGHT * NUM_PIXEL_CHARACTERISTICS , 35 , 20 , 3};
		try {
			
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
