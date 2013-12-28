package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;

import java.util.ArrayList;

import javax.swing.JComponent;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class BrainImage extends JComponent {
	public int sliceNum;
	public int lagNum;
	public int brainView;
	public ColorGradient colGrad;
	public double scale;
	public boolean useCrosshair;
	public int crosshairTransparency;
	public Color crosshairColor;
	public boolean useLabel;
	public int labelTransparency;
	public Color labelColor;
	
	public ArrayList<Integer> mCoordinates = null;
	
	public ResultModel mResultModel = null;

	//rot90times refers to number of 90 degree rotations in the CLOCKWISE direction.
	public int rot90times;
	
	private int xCrosshair = -1;
	private int yCrosshair = -1;
	
	protected BufferedImage image;
	
	protected boolean mDrawOutlineBox = true;
	public String type;
	public int lvNum;
	public String file;
	
	
	public BrainImage(ResultModel model,
			ArrayList<Integer> coords, String file, String type, int lvNum,
			int lagNum, int sliceNum, int brainView,
			ColorGradient colGrad, double scale, int rot90times,
			boolean useCrosshair, int crosshairTransparency,
			Color crosshairColor, boolean useLabel, int labelTransparency, Color labelColor) {
		this.file = file;
		this.type = type;
		this.lvNum = lvNum;
		this.lagNum = lagNum;
		this.sliceNum = sliceNum;
		this.brainView = brainView;
		this.colGrad = colGrad;
		this.useCrosshair = useCrosshair;
		this.crosshairTransparency = crosshairTransparency;
		this.crosshairColor = crosshairColor;
		this.useLabel = useLabel;
		this.labelTransparency = labelTransparency;
		this.labelColor = labelColor;
		
		mCoordinates = coords;
		mResultModel = model;

//		this.setBorder(new LineBorder(Color.RED, 2) );

		//the scale is altered in the scaleAndRotate function
		this.scale = 1.0;
		this.rot90times = 0;
		this.setDoubleBuffered(true);
		
		if(lvNum >= 0) {
			reInitializeImage();
			
			scaleAndRotate(scale,rot90times);
		}
		
//		this.setText(Integer.toString(sliceNum) );
	}
	
	public void scale(double scale) {
		scaleAndRotate(scale,rot90times);
	}
	
	public void rotate(int rot90times) {
		scaleAndRotate(scale,rot90times);
	}
	
	protected void scaleAndRotate(double scale, int rot90times) {
		this.scale = scale;
		this.rot90times = rot90times;

		Dimension imageSize = null;
		
		if (rot90times == 1 || rot90times == 3) {
			imageSize = new Dimension((int)(image.getHeight() * scale), (int)(image.getWidth() * scale) );
		}
		else {
			imageSize = new Dimension((int)(image.getWidth() * scale), (int)(image.getHeight() * scale) );
		}
		
		setMinimumSize(imageSize);
		setMaximumSize(imageSize);
		setPreferredSize(imageSize);

		invalidate();
	}
	
	
	public void reInitializeImage() {
		BrainData brainData = mResultModel.getBrainData(type);
		double[] bgImageData = mResultModel.getBgImageData();
		
		if (brainData == null) {
			System.err.println("WHAT IS DEAL WITH " + type);
		}
		
		
		
		int width = brainData.getWidth(brainView);
		int height = brainData.getHeight(brainView);
		int numSlices = brainData.getNumSlices(brainView);
		
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		
		Graphics2D g2 = (Graphics2D)image.getGraphics();
		
		g2.setPaint(Color.white);
		g2.fillRect(0, 0, width, height);
				
		//The anatomical background image is drawn here (underlay). 
		if (bgImageData != null && brainView == BrainData.AXIAL) {			
			int k = sliceNum * width * height;
			for (int i = 0; i != height; i++) {
				for (int j = 0; j != width; j++) {
					double anatomicalValue = bgImageData[k];
					//if(anatomicalValue != 0.0) System.out.println(anatomicalValue);
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(j, height - 1 - i, 1, 1);
					k++;
				}
			}
		} else if (bgImageData != null && brainView == BrainData.CORONAL) {
			int k = sliceNum * width;
			for (int i = 0; i != height; i++) {
				int temp = k;
				for (int j = 0; j != width; j++) {
					double anatomicalValue = bgImageData[temp];
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(j, height - 1 - i, 1, 1);
					temp++;
				}
				k += (width * numSlices);
			}
		} else if (bgImageData != null) { // brainView == BrainData.SAGITTAL
			int k = sliceNum;
			for (int i = 0; i != height; i++) {
				for (int j = 0; j != width; j++) {
					double anatomicalValue = bgImageData[k];
					g2.setColor(new Color((float) anatomicalValue, (float) anatomicalValue, (float) anatomicalValue));
					g2.fillRect(width - 1 - j, height - 1 - i, 1, 1);
					k += numSlices;
				}
			}
		}
		

//		mCoordinates = brainData.getSlices(brainView).get(sliceNum); 

		//coords should be 0-based.
		//The brain activity is drawn here (the overlay).
		for (int coord : mCoordinates) {
			int[] viewCoords = brainData.convert1DtoView(coord, brainView);
			int xVal = viewCoords[0];
			int yVal = viewCoords[1];

			double val;
			val = getValue(coord);
			
			// If a pixel's value is within threshold, it is not drawn.
			// However, if there is no background image then we draw a gray
			// pixel.
			if (bgImageData == null || !colGrad.inThreshold(val)) {
			
				g2.setColor(colGrad.getColor(val));
			
				// Draw a pixel (ie. rectangle with height/width=1)
			
				// The y-value is flipped since the image was being
				// displayed upside-down.
				// In the sagittal view, the x-value needs to be flipped as well
				// since the original image was also horizontally-flipped.
				if (brainView == BrainData.SAGITTAL) {
					g2.fillRect(width - 1 - xVal, height - 1 - yVal, 1,1);
				} else {
					g2.fillRect(xVal, height - 1 - yVal, 1,1);
				}
			}
		}
	}
	
	protected double getValue(int coord) {
		return mResultModel.getBrainData(type).getValue1D(coord, lagNum, lvNum);
	}
	
	public void paint(Graphics g) {
//		super.paint(g);
		
		Dimension imSize = getPreferredSize();
		int imageMidW = imSize.width / 2;
		int imageMidH = imSize.height / 2;
		
		Graphics2D g2d = (Graphics2D) g;
        AffineTransform rotate = AffineTransform.getQuadrantRotateInstance(rot90times, imageMidW, imageMidH);
        rotate.concatenate(AffineTransform.getScaleInstance(scale, scale) );
		g2d.drawImage(image, new AffineTransformOp(rotate, AffineTransformOp.TYPE_NEAREST_NEIGHBOR), 0, 0);
		
		
		drawLabel(g2d);
		drawCrosshair(g2d);
	}
	
	private void drawLabel(Graphics2D g2) {
		if (!useLabel) {
			return;
		}
		Dimension imageSize = getPreferredSize();
		
		Color newLabelColor = new Color(labelColor.getRed(), 
										labelColor.getGreen(),
										labelColor.getBlue(),
										labelTransparency);
		String sliceString = Integer.toString(sliceNum + 1);
		String lagString = Integer.toString(lagNum);
		g2.setColor(newLabelColor);
		g2.drawChars(sliceString.toCharArray(), 0, sliceString.length(), 4, imageSize.height - 4);
		g2.drawChars(lagString.toCharArray(), 0, lagString.length(), 4, 14);
	}
	
	private void drawCrosshair(Graphics2D g2) {
		Dimension imSize = getPreferredSize();
		int imageMidW = imSize.width / 2;
		int imageMidH = imSize.height / 2;
		
		AffineTransform xform = AffineTransform.getQuadrantRotateInstance(rot90times, imageMidW, imageMidH);
        xform.concatenate(AffineTransform.getScaleInstance(scale, scale) );
		
		if (useCrosshair) {
			// Determine if this image is selected
	        int width = mResultModel.getBrainData().getWidth(brainView);
			int height = mResultModel.getBrainData().getHeight(brainView);

			if (xCrosshair >= 0 && yCrosshair >= 0 &&
				xCrosshair < width &&
				yCrosshair < height ) {
				
				double newY = height - yCrosshair - 0.5;
				double newX = xCrosshair + 0.5;
		
				if (brainView == BrainData.SAGITTAL)
					newX = width - xCrosshair - 1;
		
		
				// Calculate the crosshair center
				Point2D center = new Point2D.Double(newX, newY);
				
				Point2D xcenter = xform.transform(center, null);
				
				// Draw two rectangles for the crosshair outline
				Color outlineColor = new Color(0, 0, 0, crosshairTransparency);
				g2.setColor(outlineColor);

				g2.fillRect((int)xcenter.getX() - 1, 0, 3, (int)imSize.getHeight() );
				g2.fillRect(0, (int)xcenter.getY() - 1, (int)imSize.getWidth(), 3);
				
				// Draw two lines for the crosshair
				Color newCrosshairColor = new Color(crosshairColor.getRed(), 
						crosshairColor.getGreen(),
						crosshairColor.getBlue(),
						crosshairTransparency);
				g2.setColor(newCrosshairColor);
				
				g2.fillRect((int)xcenter.getX(), 0, 1, (int)imSize.getHeight() );
				g2.fillRect(0, (int)xcenter.getY(), (int)imSize.getWidth(), 1);
				
				// This variable is set to false for ThreePaneBrainImages
				if (mDrawOutlineBox) {
					g2.setColor(outlineColor);
					
					// Draw an outline around the box
					g2.fillRect(0, 0, (int)imSize.getWidth() - 1, 1);
					g2.fillRect(0, (int)imSize.getHeight() - 1, (int)imSize.getWidth() - 1, 1);
					g2.fillRect(0, 0, 1, (int)imSize.getHeight() - 1);
					g2.fillRect((int)imSize.getWidth() - 1, 0, 1, (int)imSize.getHeight() - 1);
					
					g2.setColor(newCrosshairColor);
					
					g2.fillRect(1, 1, (int)imSize.getWidth() - 2, 1);
					g2.fillRect(1, (int)imSize.getHeight() - 2, (int)imSize.getWidth() - 2, 1);
					g2.fillRect(1, 1, 1, (int)imSize.getHeight() - 2);
					g2.fillRect((int)imSize.getWidth() - 2, 1, 1, (int)imSize.getHeight() - 2);
				}
			}
		}
	}
	
	public void setSelectedPixel(int x, int y) {
		xCrosshair = x;
		yCrosshair = y;

		repaint();
	}

	public void setCoordinates(ArrayList<Integer> coordinates) {
		mCoordinates = coordinates;
	}

	public BufferedImage getSaveImage() {
		BufferedImage newImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = newImage.createGraphics();
		
		// Do not draw crosshair onto saved image
        g2.drawImage(image, 0, 0, null);
        drawLabel(g2);
        g2.dispose();
        
		return newImage;
	}

	public Point2D worldToImageCoords(Point2D sourcePoint) {
		Dimension imSize = getPreferredSize();
		int imageMidW = imSize.width / 2;
		int imageMidH = imSize.height / 2;
		
		AffineTransform xform = AffineTransform.getQuadrantRotateInstance(rot90times, imageMidW, imageMidH);
        xform.concatenate(AffineTransform.getScaleInstance(scale, scale) );
		
		Point2D point = new Point2D.Double(sourcePoint.getX(), sourcePoint.getY() );
		
		Point2D imcoord = null;
		
		try {
			imcoord = xform.inverseTransform(point, null);
		}
		catch (NoninvertibleTransformException e) {
			// Not much we can do about this; it's just a rotation and scale
			e.printStackTrace();
		}
		
		return imcoord;
	}
}