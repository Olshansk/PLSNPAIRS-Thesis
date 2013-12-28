package pls.chrome.sessionprofile;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Scrollbar;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.io.File;
import java.io.IOException;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.border.Border;

import pls.chrome.result.model.BrainData;
import pls.shared.MLFuncs;

import npairs.io.NiftiIO;

@SuppressWarnings("serial")
public class BrainViewerPanel extends JPanel {
	private BrainImagePanel xPanel = null;
	private BrainImagePanel yPanel = null;
	private BrainImagePanel zPanel = null;
	
	public BrainViewerPanel() {
		xPanel = new BrainImagePanel("X Axis");
		yPanel = new BrainImagePanel("Y Axis");
		zPanel = new BrainImagePanel("Z Axis");
	    
	    JPanel canvasPanel = new JPanel();
	    canvasPanel.add(xPanel);
	    canvasPanel.add(yPanel);
	    canvasPanel.add(zPanel);

	    JPanel containerPanel = new JPanel(new BorderLayout());
	    containerPanel.add(new JLabel("If images are not in desired " +
				"orientation (ex. RAS, LAS), they must be changed manually " +
				"outside of PLS first."), BorderLayout.NORTH);
	    containerPanel.add(canvasPanel, BorderLayout.CENTER);
	    containerPanel.add(new JLabel("Vertical scrollbars control zoom, " +
				"horizontal scrollbars control slice."), BorderLayout.SOUTH);
	    
	    JPanel mainPanel = new JPanel();
	    mainPanel.add(containerPanel);
	    
	    add(mainPanel);
	}
	
	/**
	 * Loads a nifti brain image and tells it the image panels to display
	 * that image.
	 * @param dir The directory the image is located in.
	 * @param file The name of the image.
	 */
	public boolean loadImage(String dir, String file) {
		removeImages();
		try {
			int[] dims = NiftiIO.getVolDims3D(dir + File.separator + file);
			double[] brainData = NiftiIO.readNiftiData(dir + File.separator + file, 0);
			brainData = MLFuncs.normalize(brainData);
			
			xPanel.loadImage(dims, brainData);
			yPanel.loadImage(dims, brainData);
			zPanel.loadImage(dims, brainData);
		} catch (IOException ioex) {
			JOptionPane.showMessageDialog(this,
					"Attemping to load images into the " +
					"'Sample Image Orientation' viewer failed.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}
		return true;
	}
	
	/**
	 * Asks all image panels to remove their images.
	 */
	public void removeImages() {
		xPanel.removeImage();
		yPanel.removeImage();
		zPanel.removeImage();
	}
}

@SuppressWarnings("serial")
final class BrainImagePanel extends JPanel implements AdjustmentListener {
    public Scrollbar sliceSelector = new Scrollbar(Scrollbar.HORIZONTAL);
    public Scrollbar magnifier = new Scrollbar(Scrollbar.VERTICAL, 1, 1, 1, 10);
    private String title = null;
    private JPanel canvasPanel = new JPanel(new BorderLayout());
    PreviewBrainImage[] images = null;
    int currentSlice = 0;
    
    int width = 0, height = 0, numSlices = 0, view = -1;
    
	public BrainImagePanel(String title) {
		this.title = title;
	    
	    canvasPanel.add(sliceSelector, BorderLayout.SOUTH);
	    canvasPanel.add(magnifier, BorderLayout.EAST);
	    
	    Border border = BorderFactory.createTitledBorder(title);
	    canvasPanel.setBorder(border);
	    add(canvasPanel);
	    
	    
	    
	    canvasPanel.setPreferredSize(new Dimension(300, 300));
	    
	    magnifier.setMinimum(1);
	    magnifier.setMaximum(40 + magnifier.getVisibleAmount() );
	}
	
	/**
	 * Removes the images and prevents adjustment events.
	 */
	public void removeImage() {
		sliceSelector.removeAdjustmentListener(this);
	    magnifier.removeAdjustmentListener(this);
	    
	    if (images != null) {
	    	canvasPanel.remove(images[currentSlice]);
	    }
	    currentSlice = 0;
	}
	
	/**
	 * Creates an array of PreviewBrainImages, initializing them with the
	 * provided image.
	 * @param dims The dimensions of the images
	 * @param data The images themselves
	 */
	public void loadImage(int[] dims, double[] data) {
		
		if(title.equals("X Axis")) {
			width = dims[0];
			height = dims[1];
			numSlices = dims[2];
			view = BrainData.AXIAL;
		} else if(title.equals("Y Axis")) {
			width = dims[1];
			height = dims[2];
			numSlices = dims[0];
			view = BrainData.SAGITTAL;
		} else if(title.equals("Z Axis")) {
			width = dims[0];
			height = dims[2];
			numSlices = dims[1];
			view = BrainData.CORONAL;
		}
		
		images = new PreviewBrainImage[numSlices];
		for (int i = 0; i < numSlices; ++i) {
			images[i] = new PreviewBrainImage(data, i, view, width, height, numSlices, 1.0);
		}
		canvasPanel.add(images[currentSlice], BorderLayout.CENTER);
		
		sliceSelector.setMinimum(0);
		sliceSelector.setMaximum(numSlices + sliceSelector.getVisibleAmount() );
	    
	    sliceSelector.addAdjustmentListener(this);
	    magnifier.addAdjustmentListener(this);
	    
	    magnifier.setValue(20);
	    this.adjustmentValueChanged(new AdjustmentEvent(magnifier, 0, 0, 0) );
	}

	/**
	 * Listens for adjustment events from the slice selector and the magnifier.
	 * Changes the current image or change its scale.  In either case, causes
	 * the current image to redraw.
	 */
	@Override
	public void adjustmentValueChanged(AdjustmentEvent arg) {
		if (arg.getSource() == sliceSelector) {
			canvasPanel.remove(images[currentSlice]);
			int index = Math.min(arg.getValue(), images.length - 1);
			canvasPanel.add(images[index], BorderLayout.CENTER);
			currentSlice = index;
		} else if (arg.getSource() == magnifier) {
			double newScale = Math.max(256.0 / width, 1.0) * (magnifier.getValue() / 20.0);
			for(PreviewBrainImage image : images) {
				image.scale(newScale);
			}
		}
		
		images[currentSlice].invalidate();
		invalidate();
		((JPanel)getParent()).invalidate();
		((JPanel)getParent()).revalidate();
		getParent().repaint();
	}
}
