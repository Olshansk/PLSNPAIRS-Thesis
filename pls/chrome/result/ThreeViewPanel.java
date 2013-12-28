package pls.chrome.result;

import java.awt.Color;
import java.awt.Dimension;

import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Point2D;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;

import pls.chrome.result.blvplot.ColorGradient;
import pls.chrome.result.blvplot.NumbersPanel;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.brainimageproperties.BackgroundImageEvent;
import pls.chrome.result.controller.observer.brainimageproperties.BrainImagePropertiesObserver;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseCrosshairEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseDescriptionLabelsEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseLabelsEvent;
import pls.chrome.result.controller.observer.colourscale.ColourScaleEvent;
import pls.chrome.result.controller.observer.colourscale.ColourScaleObserver;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class ThreeViewPanel extends JPanel implements AdjustmentListener,
MouseListener, MouseMotionListener, BrainImagePropertiesObserver,
SelectionObserver, ColourScaleObserver, DataChangeObserver {
	private final static int AXIAL_IMAGE_WIDTH = 320;

	private ThreePaneBrainImage[] brainImages = null;
	private JScrollBar[] mImageScrollBars = null;
	private JLabel[] mImageLabels = null;

	private NumbersPanel mNumbersPanel = null;
	
	private GeneralRepository mRepository = null;
	
	private boolean useCrosshair = true;
	private boolean useLabel = true;
	
	// The default crosshair transparency value is 255.
	private int crosshairTransparency = 255;
	
	// The default crosshair color is green.
	private Color crosshairColor = Color.GREEN;
	
	// The default label transparency value is 255.
	private int labelTransparency = 255;
	
	// The default label color is magenta.
	private Color labelColor = Color.MAGENTA;

	private double scale = 5.0;

	private int axialX = 0;
	private int sagittalX = 0;
	private int coronalX = 0;

	public JScrollPane scroller = null;
	protected ColorGradient colorGradient = null;

	public JPanel scrollerInnerPanel = null;

	public ThreeViewPanel(GeneralRepository repository) {
		mRepository = repository;
		ResultModel model = repository.getGeneral();
		BrainData brainData = model.getBrainData();
		
		repository.getPublisher().registerObserver(this);
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		mNumbersPanel = new NumbersPanel(repository);
		add(mNumbersPanel);
		
		scrollerInnerPanel = new JPanel();
		scrollerInnerPanel.setLayout(new BoxLayout(scrollerInnerPanel, BoxLayout.X_AXIS));

		double[] colourStuff = brainData.getColourScaleModel().getColourScale();
		colorGradient = new ColorGradient(colourStuff[0], colourStuff[1], colourStuff[2]);

		ResultsCommandManager.mThreeView = this;

		brainImages = new ThreePaneBrainImage[3];

		initializeImageLabels();
		constructImages();
		
		scroller = new JScrollPane(scrollerInnerPanel);
		scroller.setAlignmentX(JScrollPane.LEFT_ALIGNMENT);
		add(scroller);
		
		updateSelection();
	}

	/**
	 * Creates the scroll bars and sets them to the right size. Adds this window
	 * as an adjustment listener.
	 */
	private void initializeScrollBars() {
		ResultModel model = mRepository.getGeneral();
		BrainData brainData = model.getBrainData();
		
		mImageScrollBars = new JScrollBar[3];

		mImageScrollBars[0] = new JScrollBar(JScrollBar.HORIZONTAL);
		mImageScrollBars[1] = new JScrollBar(JScrollBar.HORIZONTAL);
		mImageScrollBars[2] = new JScrollBar(JScrollBar.HORIZONTAL);

		mImageScrollBars[0].setMaximum(brainData.getNumSlices(BrainData.AXIAL)
				+ mImageScrollBars[0].getVisibleAmount() - 1);
		mImageScrollBars[1].setMaximum(brainData.getNumSlices(BrainData.SAGITTAL)
				+ mImageScrollBars[1].getVisibleAmount() - 1);
		mImageScrollBars[2].setMaximum(brainData.getNumSlices(BrainData.CORONAL)
				+ mImageScrollBars[2].getVisibleAmount() - 1);

		mImageScrollBars[0].addAdjustmentListener(this);
		mImageScrollBars[1].addAdjustmentListener(this);
		mImageScrollBars[2].addAdjustmentListener(this);
	}

	/**
	 * Creates labels for the 3 images.
	 */
	private void initializeImageLabels() {
		mImageLabels = new JLabel[3];

		mImageLabels[0] = new JLabel("Axial: Y vs X");
		mImageLabels[1] = new JLabel("Sagittal: Z vs Y");
		mImageLabels[2] = new JLabel("Coronal: Z vs X");
	}

	/**
	 * Removes the previous items added to the scroll pane, creates new images
	 * based on the current brain data and adds the images and their associated
	 * widgets to the scroll pane.
	 */
	public void constructImages() {
		removeAllImages();
		
		ResultModel model = mRepository.getGeneral();
		scale = (double) AXIAL_IMAGE_WIDTH / (double) model.getBrainData().getWidth(BrainData.AXIAL);
		
		String resultFileName = mRepository.getSelectedResultFile();
		
		
		
		BrainData brainData = model.getBrainData();
		
		int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
		axialX = selectedVoxel[2] - 1;
		sagittalX = selectedVoxel[0] - 1;
		coronalX = selectedVoxel[1] - 1;
		int lag = selectedVoxel[3];

		brainImages[0] = new ThreePaneBrainImage(model,
				brainData.getSlices(BrainData.AXIAL).get(axialX), resultFileName,
				axialX, lag, BrainData.AXIAL, colorGradient, scale, 0,
				useCrosshair, crosshairTransparency, crosshairColor, 
				useLabel, labelTransparency, labelColor);
		brainImages[0].scale(5.0);

		brainImages[1] = new ThreePaneBrainImage(model,
				brainData.getSlices(BrainData.SAGITTAL).get(sagittalX), resultFileName,
				sagittalX, lag, BrainData.SAGITTAL, colorGradient, scale, 0,
				useCrosshair, crosshairTransparency, crosshairColor, 
				useLabel, labelTransparency, labelColor);

		brainImages[2] = new ThreePaneBrainImage(model,
				brainData.getSlices(BrainData.CORONAL).get(coronalX), resultFileName,
				coronalX, lag, BrainData.CORONAL, colorGradient, scale, 0,
				useCrosshair, crosshairTransparency, crosshairColor, 
				useLabel, labelTransparency, labelColor);

		initializeScrollBars();

		JPanel panelAxial = addImage(0);
		JPanel panelSagittal = addImage(1);
		JPanel panelCoronal = addImage(2);
		
		JPanel firstColumn = new JPanel();
		firstColumn.setLayout(new BoxLayout(firstColumn, BoxLayout.Y_AXIS));
		firstColumn.setAlignmentY(JPanel.TOP_ALIGNMENT);
		firstColumn.add(panelAxial);
		firstColumn.add(panelCoronal);
		
		JPanel secondColumn = new JPanel();
		secondColumn.setLayout(new BoxLayout(secondColumn, BoxLayout.Y_AXIS));
		secondColumn.setAlignmentY(JPanel.TOP_ALIGNMENT);
		secondColumn.add(panelSagittal);
		
		scrollerInnerPanel.add(firstColumn);
		scrollerInnerPanel.add(secondColumn);
		
//		int width = firstColumn.getPreferredSize().width + secondColumn.getPreferredSize().width;
//		int height = Math.max(firstColumn.getPreferredSize().height, secondColumn.getPreferredSize().height);

		// Add mouse listeners to the brain images
		for (ThreePaneBrainImage im : brainImages) {
			im.addMouseListener(this);
			im.addMouseMotionListener(this);
		}
	}
	
	public void refreshImages() {
		constructImages();
	}
	
	/**
	 * Reconstructs the images.
	 */
	public void reInitializeImages() {
		for (ThreePaneBrainImage image : brainImages) {
			image.reInitializeImage();
		}
	}
	
	/**
	 * Updates the boolean value which determines if crosshairs are to
	 * be drawn and reconstructs the images.
	 */
	public void updateUseCrosshair() {
		useCrosshair = mRepository.getImagePropertiesModel().isCrosshairEnabled();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.useCrosshair = useCrosshair;
//			image.reInitializeImage();
			image.repaint();
		}
	}
	
	/**
	 * Updates the transparency value of the crosshairs to be drawn and 
	 * reconstructs the images.
	 */
	public void updateCrosshairTransparency() {
		crosshairTransparency = mRepository.getImagePropertiesModel().getCrosshairTransparency();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.crosshairTransparency = crosshairTransparency;
			image.reInitializeImage();
		}
	}
	
	/**
	 * Updates the color of the crosshairs to be drawn and reconstructs
	 * the images.
	 */
	public void updateCrosshairColor() {
		crosshairColor = mRepository.getImagePropertiesModel().getCrosshairColour();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.crosshairColor = crosshairColor;
			image.reInitializeImage();
		}
	}
	
	/**
	 * Updates the boolean value which determines if labels for lags
	 * and slices are to be displayed and reconstructs the images.
	 */
	public void updateUseLabels() {
		useLabel = mRepository.getImagePropertiesModel().labelsEnabled();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.useLabel = useLabel;
			image.repaint();
		}
	}
	
	/**
	 * Updates the transparency value of the lag/slice labels to be drawn and 
	 * reconstructs the images.
	 */
	public void updateLabelsTransparency() {
		labelTransparency = mRepository.getImagePropertiesModel().getLabelsTransparency();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.labelTransparency = labelTransparency;
			image.repaint();
		}
	}
	
	/**
	 * Updates the color of the lag/slice labels to be drawn and reconstructs
	 * the images.
	 */
	public void updateLabelsColor() {
		labelColor = mRepository.getImagePropertiesModel().getLabelsColour();
		
		for (ThreePaneBrainImage image : brainImages) {
			image.labelColor = labelColor;
			image.repaint();
		}
	}

	/**
	 * Removes all objects from the inner panel
	 */
	private void removeAllImages() {
		if (brainImages != null) {
			for (ThreePaneBrainImage im : brainImages) {
				if (im != null) {
					im.removeMouseListener(this);
					im.removeMouseMotionListener(this);
				}
			}
		}
		scrollerInnerPanel.removeAll();
	}

	/**
	 * Adds the image, label and scroll bar at index i to the scroll pane
	 */
	private JPanel addImage(int i) {
		JPanel panel = new JPanel();
		BoxLayout layout = new BoxLayout(panel, BoxLayout.Y_AXIS);
		panel.setLayout(layout);
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);

		panel.add(mImageLabels[i]);
		panel.add(brainImages[i]);
		panel.add(mImageScrollBars[i]);

		Dimension panelSize = new Dimension(brainImages[i].getPreferredSize().width, brainImages[i].getPreferredSize().height + mImageLabels[i].getPreferredSize().height + mImageScrollBars[i].getPreferredSize().height);
		panel.setPreferredSize(panelSize);
		panel.setMaximumSize(panelSize);
		
		return panel;
	}

	/**
	 * Listener for scroll bar adjustments. Causes the images to be
	 * reinitialized.
	 */
	public void adjustmentValueChanged(AdjustmentEvent event) {
		if (event.getSource() == mImageScrollBars[0]
				|| event.getSource() == mImageScrollBars[1]
				|| event.getSource() == mImageScrollBars[2]) {

			int lag = mRepository.getGeneral().getSelectionModel().getSelectedVoxel()[3];
			axialX = mImageScrollBars[0].getValue();
			sagittalX = mImageScrollBars[1].getValue();
			coronalX = mImageScrollBars[2].getValue();

			ResultsCommandManager.selectVoxel(lag, sagittalX + 1,
					coronalX + 1, axialX + 1);
		}
	}

	public void updateColourScale() { 
		double[] colours = null;
		
		if (mRepository.getUseGlobalScale() ) {
			colours = mRepository.getGlobalColourScale();
		}
		else {
			colours = mRepository.getGeneral().getBrainData().getColourScaleModel().getColourScale();
		}
		
		double max = colours[0];
		double min = colours[1];
		double threshold = colours[2];
		
		colorGradient = new ColorGradient(max, min, threshold);

		brainImages[0].colGrad = colorGradient;
		brainImages[0].setSelectedPixel(sagittalX, coronalX);
		brainImages[0].reInitializeImage();

		brainImages[1].colGrad = colorGradient;
		brainImages[1].setSelectedPixel(coronalX, axialX);
		brainImages[1].reInitializeImage();

		brainImages[2].colGrad = colorGradient;
		brainImages[2].setSelectedPixel(sagittalX, axialX);
		brainImages[2].reInitializeImage();
	}
	
	public void reInitialize() {
		String filename = mRepository.getSelectedResultFile();
		ResultModel model = mRepository.getGeneral();
		String dataName = model.getSelectedDataType();
		int lvNum = model.getBrainData().getLv();
		
		for (ThreePaneBrainImage bi : brainImages) {
			bi.mResultModel = model;
			bi.file = filename;
			bi.type = dataName;
			bi.lvNum = lvNum;
		}

		updateColourScale();
	}

	public void updateSelection() {
		ResultModel model = mRepository.getGeneral();
		BrainData brainData = model.getBrainData();

		//Selected voxel is 1-based.
		int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
		int x = selectedVoxel[0];
		int y = selectedVoxel[1];
		int z = selectedVoxel[2];
		int lag = selectedVoxel[3];
		
		int[] dimensions = brainData.getDimensions();
		// If values are legal, scroll all boxes to the right spot and
		// select the right pixels
		if (z > 0 && z <= dimensions[3]
				&& x > 0 && x <= dimensions[0]
				&& y > 0 && y <= dimensions[1]) {
			axialX = z - 1;
			sagittalX = x - 1;
			coronalX = y - 1;
	
			for (JScrollBar sbar : mImageScrollBars) {
				sbar.removeAdjustmentListener(this);
			}
			mImageScrollBars[0].setValue(axialX);
			mImageScrollBars[1].setValue(sagittalX);
			mImageScrollBars[2].setValue(coronalX);
			for (JScrollBar sbar : mImageScrollBars) {
				sbar.addAdjustmentListener(this);
			}
	
			brainImages[0].sliceNum = axialX;
			brainImages[1].sliceNum = sagittalX;
			brainImages[2].sliceNum = coronalX;
			
			brainImages[0].lagNum = lag;
			brainImages[1].lagNum = lag;
			brainImages[2].lagNum = lag;
	
			brainImages[0].setCoordinates(brainData.getSlices(BrainData.AXIAL).get(axialX) );
			brainImages[1].setCoordinates(brainData.getSlices(BrainData.SAGITTAL).get(sagittalX) );
			brainImages[2].setCoordinates(brainData.getSlices(BrainData.CORONAL).get(coronalX) );

			brainImages[0].setSelectedPixel(x - 1, y - 1);
			brainImages[0].reInitializeImage();
//			brainImages[0].repaint();
			brainImages[1].setSelectedPixel(y - 1, z - 1);
			brainImages[1].reInitializeImage();
//			brainImages[1].repaint();
			brainImages[2].setSelectedPixel(x - 1, z - 1);
			brainImages[2].reInitializeImage();
//			brainImages[2].repaint();
		}

		double value;
		value = brainData.getValue3D(x - 1, y - 1, z - 1, lag);
				
		mNumbersPanel.updateBrainValue(value);
		mNumbersPanel.refreshPanel();
	}

	// Mouse listeners for when we click on an image
	public void mouseClicked(MouseEvent e) {
		ThreePaneBrainImage bImage = (ThreePaneBrainImage) e.getSource();
		Point2D sourcePoint = new Point2D.Double(e.getX(), e.getY() );
		Point2D coordinates = bImage.worldToImageCoords(sourcePoint);
		selectPixel(bImage, (int)coordinates.getX(), (int)coordinates.getY() );
	}

	public void mouseDragged(MouseEvent e) {
		ThreePaneBrainImage bImage = (ThreePaneBrainImage) e.getSource();
		Point2D sourcePoint = new Point2D.Double(e.getX(), e.getY() );
		Point2D coordinates = bImage.worldToImageCoords(sourcePoint);
		selectPixel(bImage, (int)coordinates.getX(), (int)coordinates.getY() );
	}
	
//	public void repaint() {
//		super.repaint();
//	}

	public void mousePressed(MouseEvent e) {
	}

	public void mouseReleased(MouseEvent e) {
	}

	public void mouseEntered(MouseEvent e) {
	}

	public void mouseExited(MouseEvent e) {
	}

	public void mouseMoved(MouseEvent e) {
	}

	private void selectPixel(ThreePaneBrainImage bImage, int xCoord, int yCoord) {
		ResultModel model = mRepository.getGeneral();
		int lag = model.getSelectionModel().getSelectedVoxel()[3];
		
		BrainData brainData = model.getBrainData();
		
		int width = brainData.getWidth(bImage.brainView);
		int height = brainData.getHeight(bImage.brainView);

		// The y-value is flipped since the image was being displayed
		// upside-down. Because of this, the y-value needs to be flipped
		// for retrieving the brain value as well.
		yCoord = height - 1 - yCoord;

		// In the sagittal view, the x-value needs to be flipped as well
		// since the original image was also horizontally-flipped.
		if (bImage.brainView == BrainData.SAGITTAL) {
			xCoord = width - 1 - xCoord;
		}

		int xSelect = xCoord, ySelect = yCoord, zSelect = bImage.sliceNum;

		if (bImage.brainView == BrainData.SAGITTAL) {
			xSelect = bImage.sliceNum;
			ySelect = xCoord;
			zSelect = yCoord;
		} else if (bImage.brainView == BrainData.CORONAL) {
			ySelect = bImage.sliceNum;
			zSelect = yCoord;
		}

		ResultsCommandManager.selectVoxel(lag, xSelect + 1, ySelect + 1,
				zSelect + 1);
	}

	@Override
	public void notify(BackgroundImageEvent e) {
		constructImages();
	}

	@Override
	public void notify(CrosshairColourEvent e) {
		updateCrosshairColor();
	}

	@Override
	public void notify(CrosshairTransparencyEvent e) {
		updateCrosshairTransparency();
	}

	@Override
	public void notify(LabelsColourEvent e) {
		updateLabelsColor();
	}

	@Override
	public void notify(LabelsTransparencyEvent e) {
		updateLabelsTransparency();
	}

	@Override
	public void notify(UseCrosshairEvent e) {
		updateUseCrosshair();
	}

	@Override
	public void notify(UseDescriptionLabelsEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notify(UseLabelsEvent e) {
		updateUseLabels();
	}

	@Override
	public void notify(Event e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notify(SelectedDataTypeChangedEvent e) {
		reInitialize();
	}

	@Override
	public void notify(SelectedLvChangedEvent e) {
		reInitialize();
	}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {
		reInitialize();
	}

	@Override
	public void notify(SelectionEvent e) {
		updateSelection();
	}

	@Override
	public void notify(ColourScaleEvent e) {
		updateColourScale();
	}

	@Override
	public void notify(FlipVolumeEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notify(LoadedVolumesEvent e) {
		constructImages();
	}

	@Override
	public void notify(InvertedLvEvent e) {
		reInitialize();
	}
}