import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class ImageBrain extends Brain {
	
	private final int PRE_DATA_LINES = 1;
	
	private int[][] rgb; // Defined in loadData, NOT in constructor
	private int[] unrolledRgb; // Defined in loadData, NOT in constructor

	/**
	 * 
	 * @param layerSizes The number of nodes in each layer of the network. Index 0 is the input layer, and the last number is the output layer.
	 * @throws BrainException
	 */
	public ImageBrain(int[] layerSizes) throws BrainException {

		super(layerSizes);
		
	}
	
	/**
	 * Loads and unrolls images into a 2D array: 1st dimension is which image, 2nd is the image's pixel values.
	 * @param information 0:dataFileName 1:imageDirectory 2:imageWidth 3:imageHeight 4:whiteBackground(in int form: 0=false, else=true)
	 * @param dataFileName The file path and name of data, formatted with 1st line as all labels, and after, one entry per line: input,correctOutput
	 * @param imageDirectory The file path of the folder in which all of the images lie.
	 * @param imageWidth The width of ADJUSTED (downsized) image, at which all images are compared.
	 * @param imageHeight The height of ADJUSTED (downsized) image, at which all images are compared.
	 * @param whiteBackground Whether the object in question is against a white background in all cases.
	 */
	public void loadData(String[] information) throws BrainException {
		
		loadData(information[0] , information[1] , Integer.parseInt(information[2]) , Integer.parseInt(information[3]) , Integer.parseInt(information[4]) != 0);
		
	}
	
	/**
	 * Loads and unrolls images into a 2D array: 1st dimension is which image, 2nd is the image's pixel values.
	 * @param dataFileName The file path and name of data, formatted with 1st line as all labels, and after, one entry per line: input,correctOutput
	 * @param imageDirectory The file path of the folder in which all of the images lie.
	 * @param imageWidth The width of ADJUSTED (downsized) image, at which all images are compared.
	 * @param imageHeight The height of ADJUSTED (downsized) image, at which all images are compared.
	 * @param whiteBackground Whether the object in question is against a white background in all cases.
	 * @throws BrainException 
	 */
	private void loadData(String dataFileName , String imageDirectory , int imageWidth , int imageHeight , boolean whiteBackground) throws BrainException {
		
		// Read in data from file -> dataFileLines
		super.dataFileLines = super.fileToLines(dataFileName);
		
		// Now, convert image dataFileLines into Test Cases (input, output):
		int height = imageHeight;
		int width = imageWidth;
		int imageSize = height * width;
		// Because an image is is represented with height as the first dimension of a matrix, 
		// width as the second, this divides 1st dimension / 2nd dimension:
		double dimRatio = height / width;
		
		// Re-set brain's test case size to actual number of test cases
		int numImages = super.dataFileLines.length - PRE_DATA_LINES;
		super.cases = new TestCase[numImages];
		
		// Ensure the image directory ends with '/'
		if (imageDirectory.charAt(imageDirectory.length() - 1) != '/')
			imageDirectory = imageDirectory + '/';
		
		// Read all possible labels from file, all on 1st line separated by commas (END WITH A COMMA)
		int start = 0 , nextLabelLength , x = 0;
		while ((nextLabelLength = super.dataFileLines[0].substring(start).indexOf(',')) != -1){
			
			int end = start + nextLabelLength;
			super.labels[x] = super.dataFileLines[0].substring(start, end);
			start = end + 1;
			x++;
			
		}
		
		// Read the image data file:
		for (x = PRE_DATA_LINES ; x <= numImages ; x++) {
			
			System.out.println("Reading image " + (x - PRE_DATA_LINES) + " / " + numImages);
			
			int split = super.dataFileLines[x].indexOf(',');
			
			// Read in image file name
			String imageName = super.dataFileLines[x].substring(0 , split);
			// Append file name to image directory for full file path
			imageName = imageDirectory + imageName;
			
			// Read in label
			String label = super.dataFileLines[x].substring(split + 1);
			
			// Read in pixel data
			BufferedImage image = null;
			try {
				
			    image = ImageIO.read(new File(imageName));
			    
			} catch (IOException e) {
				
				// Invalid image, downsize test cases array and try the next one
				TestCase[] temp = super.cases;
            	super.cases = new TestCase[super.cases.length - 1];
            	System.arraycopy(temp , 0 , super.cases , 0 , temp.length);
				continue;
				
			}
			
			// RESIZE so not cripplingly large
			image = ImageProcessor.resize(image , width , height , Image.SCALE_SMOOTH);
			
			int[] argb = image.getRGB(0 , 0 , image.getWidth() , image.getHeight() , null , 0 , image.getWidth());
			
			// MUST ASSUME pixel data is in correct orientation, dimensions <--- IMPROVE THIS
			// Extract r,g,b components from TYPE_INT_ARGB:
			// First, convert int into its four component bytes
			byte[][] rgbBytes = new byte[argb.length][4];
			for (int i = 0 ; i < argb.length ; i++)
				rgbBytes[i]= ByteBuffer.allocate(4).putInt(argb[i]).array();
			
			// Now, convert these bytes to ints and unroll into 1D array: reds then greens then blues
			rgb = new int[3][rgbBytes.length];
			unrolledRgb = new int[3 * rgb[0].length];
			for (int i = 0 ; i < rgbBytes.length ; i++) {
				
				unrolledRgb[i] = rgb[0][i] = rgbBytes[i][1] & 0xFF;
				unrolledRgb[rgb[0].length + i] = rgb[1][i] = rgbBytes[i][2] & 0xFF;
				unrolledRgb[2 * rgb[0].length + i] = rgb[2][i] = rgbBytes[i][3] & 0xFF;
				
			}
			
			// Now, must PROCESS the images to make them as primitive as possible before use in the neural network:
			
			// Remove the white background if presence of one is mandated
			if (whiteBackground)
				unrolledRgb = ImageProcessor.removeBackground(unrolledRgb , rgb.length);
			
			// Do other processing - filtering (Gaussian, Laplacian?), edge detection, etc
			// Modify super's input layer size to match size of modified data
			// Ideas:
			// - edge detect then scale all images to same horizontal scale? Make processed dimensions square to accommodate out of bounds?
			
			// Convert to double array, put it all into a TestCase object and add it to the brain's list
			super.cases[x - PRE_DATA_LINES] = new TestCase(Arrays.stream(unrolledRgb).asDoubleStream().toArray() , label , imageName);
			
		}
		
	}
	
	
	
	private static class ImageProcessor {
		
		private static int[] removeBackground(int[] unrolledRgb , int gap) {
			
			// Remove background in unrolledRgb -- see image processing lecture iii
			
			// Edge detection using internal & external gradient
			
			return unrolledRgb;
			
		}
		
		public static BufferedImage resize(BufferedImage image , int newWidth , int newHeight , int SCALE) {
			
			Image scaledImage = image.getScaledInstance(newWidth, newHeight, SCALE);
			BufferedImage bScaledImage = new BufferedImage(newWidth , newHeight , BufferedImage.TYPE_INT_ARGB);
			
			Graphics2D g = bScaledImage.createGraphics();
			g.drawImage(scaledImage , 0 , 0 , null);
			g.dispose();
			
			return bScaledImage;
			
		}
		
	}

}
