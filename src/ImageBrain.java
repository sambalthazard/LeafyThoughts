import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.IntSummaryStatistics;

import javax.imageio.ImageIO;

public class ImageBrain extends Brain {
	
	private static final int PRE_DATA_LINES = 1;
	public static final double RED_GRAY_COEFFICIENT = .21;
	public static final double GREEN_GRAY_COEFFICIENT = .72;
	public static final double BLUE_GRAY_COEFFICIENT = .07;
	public static final boolean UNROLLED_VERTICAL = false;
	
	// MODIFIABLE FINALS
	public static final int EDGE_DETECT_STRUCTURING_ELEMENT_SIZE = 3; // MUST be odd
	
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
	public ImageBrain(int[] layerSizes , double trainingRatio , double validationRatio , double testRatio) throws BrainException {

		super(layerSizes , trainingRatio , validationRatio , testRatio);
		
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
			
			System.out.println("Reading image " + (x - PRE_DATA_LINES + 1) + " / " + numImages);
			
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
			
			// Extract fullsize grayscale image for processing; will be downsized and the larger array garbaged before next iteration
			// See immediately below for near-identical operation with full comments
			int origWidth = image.getWidth();
			int origHeight = image.getHeight();
			int[] fullsizeGray = new int[origWidth * origHeight];
			int[] argb = image.getRGB(0 , 0 , origWidth , origHeight , null , 0 , origWidth);
			
			argb = image.getRGB(0 , 0 , origWidth , origHeight , null , 0 , origWidth);
			
			byte[][] rgbBytes = new byte[argb.length][4];
			for (int i = 0 ; i < argb.length ; i++)
				rgbBytes[i]= ByteBuffer.allocate(4).putInt(argb[i]).array();
			
			for (int i = 0 ; i < rgbBytes.length ; i++)
				fullsizeGray[i] = (int) ((rgbBytes[i][1] & 0xFF) * RED_GRAY_COEFFICIENT +
											(rgbBytes[i][2] & 0xFF) * GREEN_GRAY_COEFFICIENT +
											(rgbBytes[i][3] & 0xFF) * BLUE_GRAY_COEFFICIENT);
			
			// RESIZE so not cripplingly large for rgb extraction
			image = ImageProcessor.resize(image , width , height , Image.SCALE_SMOOTH);
			
			// MUST ASSUME pixel data is in correct orientation, dimensions <--- IMPROVE THIS
			// Extract r,g,b components from TYPE_INT_ARGB:
			// First, convert int into its four component bytes
			rgbBytes = new byte[argb.length][4];
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
			
			// Detect edges and put into binary (0 or 1) 2D array; via morphological gradient -> adaptive thresholding
			int[] edges = ImageProcessor.detectBinaryEdges(fullsizeGray , origWidth , origHeight , UNROLLED_VERTICAL);
			
			// Resize fullsize grayscale down to required size for NN
			int[] processedGray = ImageProcessor.resizeGrayscaleBilinear(edges , origWidth, origHeight, width, height);
			
			// Do other processing - filtering (Gaussian, Laplacian?), edge detection, etc
			// Modify super's input layer size to match size of modified data
			// Ideas:
			// - edge detect then scale all images to same horizontal scale? Make processed dimensions square to accommodate out of bounds?
			
			// Convert to double array, feature scale, put it all into a TestCase object and add it to the brain's list
			double[] processedImageDoubleArray = Arrays.stream(unrolledRgb).asDoubleStream().toArray();
			double[] featureScaled = Brain.featureScale(processedImageDoubleArray);
			super.cases[x - PRE_DATA_LINES] = new TestCase(featureScaled , label , imageName);
			
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
		
		public static int[] resizeGrayscaleBilinear(int[] intensities , int initWidth , int initHeight , int newWidth , int newHeight) {

		    float xDiff , yDiff;
		    int topLeft, topRight, botLeft, botRight , x, y , index , intensity;
		    
		    int[] result = new int[newWidth * newHeight];
		    int resultIndex = 0;
		    
		    float xRatio = ((float)(initWidth - 1)) / newWidth;
		    float yRatio = ((float)(initHeight - 1)) / newHeight;
		    for (int i = 0 ; i < newHeight ; i++) {
		    	
		        for (int j = 0 ; j < newWidth ; j++) {
		        	
		            x = (int)(xRatio * j);
		            y = (int)(yRatio * i);
		            xDiff = (xRatio * j) - x;
		            yDiff = (yRatio * i) - y;
		            index = y * initWidth + x;

		            // Range is 0 to 255 thus bitwise AND with 0xff
		            topLeft = intensities[index] & 0xff ;
		            topRight = intensities[index + 1] & 0xff ;
		            botLeft = intensities[index + initWidth] & 0xff ;
		            botRight = intensities[index + initWidth + 1] & 0xff ;
		            
		            // Bilinearly interpolate
		            intensity = (int)(topLeft * (1 - xDiff) * (1 - yDiff)  +  topRight * (xDiff) * (1 - yDiff) +
		                    botLeft * (yDiff) *(1 - xDiff)  +  botRight * (xDiff * yDiff));

		            result[resultIndex] = intensity ;        
		            resultIndex++;
		            
		        }
		        
		    }
		    
		    return result;
		    
		}
		
		/**
		 * Detects edges in the provided grayscale image and returns as a binary (int 0 or 1) 2D array. 1 = edge pixel, 0 = not edge pixel.
		 * Returns as int rather than boolean for easier processing in the brain.
		 * @param intensities A GRAYSCALE image represented by an array of 8-bit intensities: 0-255.
		 * @param width 
		 * @param height 
		 * @param vertical 
		 * @return A binary (int 0 or 1) 2D array of edges in the image.
		 * @throws BrainException 
		 */
		public static int[] detectBinaryEdges(int[] intensities , int width , int height , boolean vertical) throws BrainException {
			
			// Ensure input is valid grayscale intensities
			// NOTE: this does not ensure the input is in fact grayscale, as it would throw no exception with an input of sequential RGB intensities.
			IntSummaryStatistics stat = Arrays.stream(intensities).summaryStatistics();
			if (stat.getMin() < 0 || stat.getMax() > 255)
			    throw new BrainException("Invalid pixel intensity; edge detection failed.");
		    
			// morphological gradient
			int[] nonbinaryEdges = detectEdges(intensities , width , height , vertical , EDGE_DETECT_STRUCTURING_ELEMENT_SIZE);
			
			// adaptive thresholding
			int[] binaryEdges = adaptiveThreshold(nonbinaryEdges);
			
			return binaryEdges;
			
		}
			
		/**
		 * Converts int array of colors listed by R,G,B into grayscale intensities.
		 * @param rgb An int array of colors, with index 0 as red, 1 as blue, 2 as green and underlying arrays are pixels' respective intensities.
		 * @return The grayscale intensities of each pixel provided in the input array.
		 */
		public static int[] toGrayscale(int[][] rgb) {
			
			int[] grayIntensities = new int[rgb[0].length];
			
			for (int i = 0 ; i < rgb[0].length ; i++) {
				
				int thisGrayIntensity = (int) (RED_GRAY_COEFFICIENT * rgb[0][i] +
									GREEN_GRAY_COEFFICIENT * rgb[1][i] +
									BLUE_GRAY_COEFFICIENT * rgb[2][i]);
				grayIntensities[i] = thisGrayIntensity;
				
			}
			
			return grayIntensities;
			
		}
		
		/**
		 * Performs morphological gradient edge detection on grayscale image pixel intensities.
		 * @param intensities The grayscale intensities of each pixel in the image.
		 * @param width 
		 * @param height 
		 * @param vertical 
		 * @param structElemDim The size of the structuring element used for morphological gradients.
		 * @return
		 */
		public static int[] detectEdges(int[] intensities , int width , int height , boolean vertical , int structElemDim) {
			
			// CODE HERE
			
		}
		
		public static int[] adaptiveThreshold(int[] intensities) {
			
			// CODE HERE
			
		}
		
	}

}
