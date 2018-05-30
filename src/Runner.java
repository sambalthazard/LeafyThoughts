
public class Runner {
	
	public static void main(String[] args) {
		
		final int IMAGE_WIDTH = 160;
		final int IMAGE_HEIGHT = 90;
		final int NUM_PIXEL_CHARACTERISTICS = 3; // R, G, and B
		
		int[] leafIDBrainLayerSizes = {IMAGE_WIDTH * IMAGE_HEIGHT * NUM_PIXEL_CHARACTERISTICS , 35 , 20 , 3};
		try {
			
			ImageBrain leafIDBrain = new ImageBrain(leafIDBrainLayerSizes);
			
			String[] leafIDInformation = {"images/leafData.txt" , "images/" , IMAGE_WIDTH + "" , IMAGE_HEIGHT + "" , "1"};
			leafIDBrain.loadData(leafIDInformation);
			leafIDBrain.train();
			
		} catch (BrainException e) {
			
			e.printStackTrace();
			
		}
		
	}
	
}
