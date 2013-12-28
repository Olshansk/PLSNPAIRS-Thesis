package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JLabel;

@SuppressWarnings("serial")
public class ColorBar extends JLabel {
	ColorGradient colGrad;
	private BufferedImage image;
	private ImageIcon icon;
	int pixelWidth,pixelHeight;
	ColorBar(ColorGradient colGrad,int pixelWidth, int pixelHeight) {
		this.colGrad = colGrad;
		this.pixelHeight = pixelHeight;
		this.pixelWidth = pixelWidth;
		initImage();
		refresh(image);
	}
	
	void initImage() {
	
		image = new BufferedImage(pixelWidth, pixelHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = (Graphics2D)image.getGraphics();
		
		double interval = (colGrad.max - colGrad.min) / (double)pixelWidth;
		for(int i = 0; i < pixelWidth; i++) {
			double simulatedValue = colGrad.min + (interval * i);
			Color c = colGrad.getColor(simulatedValue);
			g2.setPaint(c);
			
			if (i == 350) {
				g2.setPaint(Color.GREEN);
			}
			//paint a 1-pixel-high strip across corresponding to the
			//color in the gradient
			g2.fillRect(i, 0, 1, pixelHeight);
		}
	}
	
	void refresh(BufferedImage im) {
		icon = new ImageIcon(im);
		this.setIcon(icon);
	}
	
}
