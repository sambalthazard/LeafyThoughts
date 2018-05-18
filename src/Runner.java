
public class Runner {
	
	public static void main(String[] args) {
		
		int[] leafIDBrainLayerSizes = {10800 , 35 , 20 , 3};
		try {
			
			ImageBrain leafIDBrain = new ImageBrain(leafIDBrainLayerSizes);
			
			String[] leafIDInformation = {"images/leafData.txt" , "images/" , "80" , "45" , "1"};
			leafIDBrain.loadData(leafIDInformation);
			leafIDBrain.train();
			
		} catch (BrainException e) {
			
			e.printStackTrace();
			
		}
		
	}
	
}
