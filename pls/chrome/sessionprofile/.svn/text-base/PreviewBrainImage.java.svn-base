package pls.chrome.sessionprofile;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import pls.chrome.result.blvplot.BrainImage;
import pls.chrome.result.model.BrainData;

/**
 * This class is similar to other brain images but only shows values for
 * @author jacques
 *
 */
@SuppressWarnings("serial")
public class PreviewBrainImage extends BrainImage {
	private double[] brainData;
	private int width;
	private int height;
	private int numSlices;
	
	public PreviewBrainImage(double[] brainData, int sliceNum,
			int brainView, int width, int height, int numSlices,
			double scale) {
		super(null, null, null, null, -1, -1, sliceNum, brainView, null,
				scale, 0, false, -1, null,
				false, -1, null);
		
		this.brainData = brainData;
		this.width = width;
		this.height = height;
		this.numSlices = numSlices;
		
		reInitializeImage();
		// We want the image to be at least300 pixels wide.
		double newScale = Math.max(width / 300.0, 1.0);
		scaleAndRotate(newScale, 0);
	}
	
	public void reInitializeImage() {
		
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		
		Graphics2D g2 = (Graphics2D)image.getGraphics();
		
		g2.setPaint(Color.white);
		g2.fillRect(0, 0, width, height);
		
		if (brainView == BrainData.AXIAL) {			
			int k = sliceNum * width * height;
			for (int i = 0; i != height; i++) {
				for (int j = 0; j != width; j++) {
					double anatomicalValue = brainData[k];
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(j, height - 1 - i, 1, 1);
					k++;
				}
			}
		} else if (brainView == BrainData.CORONAL) {
			int k = sliceNum * width;
			for (int i = 0; i != height; i++) {
				int temp = k;
				for (int j = 0; j != width; j++) {
					double anatomicalValue = brainData[temp];
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(j, height - 1 - i, 1, 1);
					temp++;
				}
				k += (width * numSlices);
			}
		} else { // brainView == BrainData.SAGITTAL
			int k = sliceNum;
			for (int i = 0; i != height; i++) {
				for (int j = 0; j != width; j++) {
					double anatomicalValue = brainData[k];
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(width - 1 - j, height - 1 - i, 1, 1);
					k += numSlices;
				}
			}
		}
	}
}
