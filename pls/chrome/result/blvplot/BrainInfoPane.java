package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JToolBar;
import javax.swing.filechooser.FileNameExtensionFilter;

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
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.filters.FiltersObserver;
import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.controller.observer.singlebrainimageview.BrainViewEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.RotationEvent;
import pls.chrome.result.controller.observer.singlebrainimageview.SingleBrainImageViewObserver;
import pls.chrome.result.controller.observer.singlebrainimageview.ZoomEvent;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ImageWriter;
import pls.shared.GlobalVariablesFunctions;

/**
 * This is the panel that lies inside each tab of the PlotTypeTabbedPane
 * It contains the NumbersPanel and ImageMontagePanel
 */
@SuppressWarnings("serial")
public class BrainInfoPane extends JPanel implements ActionListener,
SelectionObserver, BrainImagePropertiesObserver, FiltersObserver,
ColourScaleObserver, DataChangeObserver, SingleBrainImageViewObserver, AdjustmentListener {
	public ImageMontagePanel imageMontagePanel;
	public NumbersPanel numbersPanel;
	public ColorGradient colorGradient;
	private ColorGradientScheme cgScheme;
	
	public JPanel notesPanel;
	public JPanel notesInnerPanel;
//	public int mBrainView = BrainData.AXIAL;
	public GeneralRepository mRepository = null;
	public double zoomValue = 100;
	
	// All this to make zoom work.  grrr.....
	JScrollPane montageScroller;
	double horzScrollPercent = -1;
	int horzMax = 0;
	double vertScrollPercent = -1;
	int vertMax = 0;
	boolean zoomHack = true;
	
	boolean vertDone = false;
	boolean horzDone = false;
	
	//Widgets
	public JButton zoomIn = null;
	public JButton zoomOut = null;
	public JButton rot90Button = null;
	public JButton rotNeg90Button = null;
	public JComboBox brainViewBox = null;
	public JButton saveImages = null;
	
	private String filePath = ".";
	
	public BrainInfoPane(GeneralRepository repository) {
		mRepository = repository;
		
		ResultsCommandManager.mBrainPane = this;
		repository.getPublisher().registerObserver(this);
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		
		updateWidgets();
		updateSelection();
	}
	
	public void setBrainView()
	{
		imageMontagePanel.removeAllImages();
		imageMontagePanel.constructImages();
		updateSelection();
		
//		revalidate();
//		repaint();
	}
	
	public void updateSliceFilters()
	{
		imageMontagePanel.updateSliceFilters();
		revalidate();
		repaint();
	}
	
	public void reInitializeImages(String resultFile)
	{
		imageMontagePanel.reInitializeImages(resultFile);
	}
	
	public void updateUseCrosshair()
	{
		boolean useCrosshair = mRepository.getImagePropertiesModel().isCrosshairEnabled();
		imageMontagePanel.setCrosshair(useCrosshair);
	}
	
	public void updateCrosshairTransparency()
	{
		int value = mRepository.getImagePropertiesModel().getCrosshairTransparency();
		imageMontagePanel.setCrosshairTransparency(value);
	}
	
	public void updateCrosshairColor()
	{
		Color colour = mRepository.getImagePropertiesModel().getCrosshairColour();
		imageMontagePanel.setCrosshairColor(colour);
	}
	
	public void updateUseLabels()
	{
		boolean useLabel = mRepository.getImagePropertiesModel().labelsEnabled();
		imageMontagePanel.setLabel(useLabel);
	}
	
	public void updateLabelsTransparency()
	{
		int value = mRepository.getImagePropertiesModel().getLabelsTransparency();
		imageMontagePanel.setLabelTransparency(value);
	}
	
	public void updateLabelsColor()
	{
		Color color = mRepository.getImagePropertiesModel().getLabelsColour();
		imageMontagePanel.setLabelColor(color);
	}
	
	public void updateUseDescriptionLabels() {
		boolean useLabels = mRepository.getImagePropertiesModel().descriptionLabelsEnabled();
		imageMontagePanel.setOtherLabels(useLabels);
		
		refreshPlot();
	}
	
	private void updateWidgets()
	{
		BrainData bData = mRepository.getGeneral().getBrainData();
		
		// Make the numbers panel
		if (numbersPanel != null)
			remove(numbersPanel);
		
		if (imageMontagePanel != null)
			remove(imageMontagePanel);
		
		numbersPanel = new NumbersPanel(mRepository);
		add(numbersPanel);
		numbersPanel.setMaximumSize(numbersPanel.getPreferredSize());
		
		// Make the toolbar
		JToolBar toolBar = new JToolBar();
		toolBar.setAlignmentX(JToolBar.LEFT_ALIGNMENT);
		toolBar.setFloatable(false);
		
		java.net.URL imgURL = this.getClass().getResource("/images/cw.png");
		rot90Button = new JButton(new ImageIcon(imgURL));
		rot90Button.setToolTipText("Rotate the images 90 degrees clockwise.");
		
		imgURL = this.getClass().getResource("/images/ccw.png");
		rotNeg90Button = new JButton(new ImageIcon(imgURL));
		rotNeg90Button.setToolTipText("Rotate the images 90 degrees counter-clockwise.");

		// By popular request, the placeholder zoom in/out icons are used
//		imgURL = this.getClass().getResource("/toolbarButtonGraphics/general/ZoomIn24.gif");
		imgURL = this.getClass().getResource("/images/zoomin.png");
		zoomIn = new JButton(new ImageIcon(imgURL));

//		imgURL = this.getClass().getResource("/toolbarButtonGraphics/general/ZoomOut24.gif");
		imgURL = this.getClass().getResource("/images/zoomout.png");
		zoomOut = new JButton(new ImageIcon(imgURL));

		brainViewBox = new JComboBox();
		brainViewBox.insertItemAt("Axial", BrainData.AXIAL);
		brainViewBox.insertItemAt("Sagittal",BrainData.SAGITTAL);
		brainViewBox.insertItemAt("Coronal",BrainData.CORONAL);
		brainViewBox.setSelectedIndex(BrainData.AXIAL);
		brainViewBox.setToolTipText("Select from one of the three cardinal planes to view the brain images.");
		
		saveImages = new JButton("Save Brain Slice Images",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Save24.gif")));
		saveImages.setToolTipText("Save the BrainImages below to a file.");
		
		JLabel changeColorSchemeLabel = new JLabel("Change Colour Scheme:");
		JComboBox ccbox = new JComboBox();
		for(String scheme : cgScheme.colorSchemes){
			ccbox.addItem(scheme);
		}
		ccbox.addActionListener(this);
		ccbox.setActionCommand("changeScheme");
		changeColorSchemeLabel.setLabelFor(ccbox);
		
		toolBar.add(rot90Button);
		toolBar.add(rotNeg90Button);
		toolBar.add(zoomIn);
		toolBar.add(zoomOut);
		toolBar.add(new JToolBar.Separator());
		toolBar.add(new JLabel("View: "));
		toolBar.add(brainViewBox);
		
		toolBar.add(new JToolBar.Separator());
		toolBar.add(saveImages);
		toolBar.add(new JToolBar.Separator());
		toolBar.add(new JLabel("Change Colour Scheme: "));
		toolBar.add(ccbox);
		
		toolBar.setPreferredSize(new Dimension(toolBar.getPreferredSize().width, zoomOut.getPreferredSize().height) );
		toolBar.setMinimumSize(new Dimension(toolBar.getPreferredSize().width, zoomOut.getPreferredSize().height) );
		toolBar.setMaximumSize(new Dimension(toolBar.getPreferredSize().width, zoomOut.getPreferredSize().height) );
		
		add(toolBar);
		
		// Make the montage
		double[] maxMinThresh = bData.getMaxMinThresh();
		colorGradient = new ColorGradient(maxMinThresh[0], maxMinThresh[1], maxMinThresh[2]);
		
		imageMontagePanel = new ImageMontagePanel(mRepository, colorGradient);
		
		montageScroller = new JScrollPane(imageMontagePanel);
		montageScroller.setAlignmentX(JScrollPane.LEFT_ALIGNMENT);
		
		montageScroller.getHorizontalScrollBar().addAdjustmentListener(this);
		montageScroller.getVerticalScrollBar().addAdjustmentListener(this);

		add(montageScroller);
		
		rot90Button.setActionCommand("rotate right");
		rotNeg90Button.setActionCommand("rotate left");
		zoomIn.setActionCommand("zoom in");
		zoomOut.setActionCommand("zoom out");
		brainViewBox.setActionCommand("set view");
		saveImages.setActionCommand("save brain slice images");
		
		rot90Button.addActionListener(this);
		rotNeg90Button.addActionListener(this);
		zoomIn.addActionListener(this);
		zoomOut.addActionListener(this);
		brainViewBox.addActionListener(this);
		saveImages.addActionListener(this);
		
		zoomIn.setToolTipText("Zoom in. Current zoom level: " + zoomValue + "%");
		zoomOut.setToolTipText("Zoom out. Current zoom level: " + zoomValue + "%");
	}
	
	public void updateColourScale()
	{
//		imageMontagePanel.reInitialize();
		imageMontagePanel.constructImages();
		updateSelection();
	}
	
	public void updateZoom() {
		double zoomValue = mRepository.getImagePropertiesModel().getZoom();
		
		imageMontagePanel.scaleImages(zoomValue);
		
		String zoomValueString = Double.toString(zoomValue * 100);
		int index = zoomValueString.indexOf(".");
		if (index != -1) {
			zoomValueString = zoomValueString.substring(0, index + 2);
		}
		zoomIn.setToolTipText("Zoom in. Current zoom level: " + zoomValueString + "%");
		zoomOut.setToolTipText("Zoom out. Current zoom level: " + zoomValueString + "%");
	}
	
	/**
	 * Updates the selected image by asking the model which result file,
	 * data type, lv and voxel is selected.  Deselects the previously selected
	 * image.
	 */
	public void updateSelection() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		String selectedResultFile = mRepository.getSelectedResultFile();
		ResultModel model = mRepository.getGeneral();

		//selected voxel uses 1-based indexing.
		int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
		int x = selectedVoxel[0];
		int y = selectedVoxel[1];
		int z = selectedVoxel[2];
		int lag = selectedVoxel[3];
		
		String selectedDataType = model.getSelectedDataType();
		BrainData brainData = model.getBrainData();
		int selectedLv = brainData.getLv();
		
		int xcoord = x, ycoord = y, slice = z;
		double brainVal = 0;
		
		
		if (brainView == BrainData.SAGITTAL)
		{
			slice = x;
			xcoord = y;
			ycoord = z;
		}
		
		if (brainView == BrainData.CORONAL)
		{
			slice = y;
			ycoord = z;
		}

		BrainImage newSelection = null;
		
		if (imageMontagePanel.mBrainImages.containsKey(selectedResultFile) ) {
			HashMap<String, HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>>> currFile = imageMontagePanel.mBrainImages.get(selectedResultFile);
			
			if (currFile.containsKey(selectedDataType) ) {
				HashMap<Integer, HashMap<Integer, HashMap<Integer, BrainImage>>> currType = currFile.get(selectedDataType);
				
				if (currType.containsKey(selectedLv + 1) ) {
					HashMap<Integer, HashMap<Integer, BrainImage>> currLv = currType.get(selectedLv +1);
					
					if (currLv.containsKey(lag) ) {
						HashMap<Integer, BrainImage> currLag = currLv.get(lag);
						
						if (currLag.containsKey(slice) ) {
							newSelection = currLag.get(slice);
						}
					}
				}
			}
		}
		
		if (imageMontagePanel.currentlySelected != null &&
				imageMontagePanel.currentlySelected != newSelection) {
			imageMontagePanel.currentlySelected.setSelectedPixel(-1, -1);
//			imageMontagePanel.currentlySelected.reInitializeImage();
			imageMontagePanel.currentlySelected.repaint();
		}
		
		imageMontagePanel.currentlySelected = newSelection;

		//coordinates based on 1-based indexing.
		//newSelection.sliceNum is 0-based.
		if (newSelection != null) {
			newSelection.setSelectedPixel(xcoord - 1, ycoord - 1);
			newSelection.repaint();
			brainVal = brainData.getValueView(brainView, xcoord - 1,
					ycoord - 1, newSelection.sliceNum, newSelection.lagNum);
		}
		
		numbersPanel.updateBrainValue(brainVal);
		numbersPanel.refreshPanel();
	}
	
	/**
	 * Asks the numbers panel to refresh and the image montage panel
	 * to create new images and reinitialize existing ones.
	 */
	public void refreshPlot() {
		numbersPanel.refreshPanel();
		imageMontagePanel.constructImages();
		updateSelection();
	}
	
	public void reinitializePlot() {
		numbersPanel.refreshPanel();
		imageMontagePanel.reInitialize();
		updateSelection();
	}
	
	public void actionPerformed(ActionEvent e) {
		int rotation = mRepository.getImagePropertiesModel().getRotation();
		double zoom = mRepository.getImagePropertiesModel().getZoom();
		
		if (e.getActionCommand().equals("rotate right")) {
			ResultsCommandManager.setImageRotation(rotation + 1);
//			imageMontagePanel.rotateImages(1);
		}
		else if (e.getActionCommand().equals("rotate left")) {
			ResultsCommandManager.setImageRotation(rotation - 1);
//			imageMontagePanel.rotateImages(3);
		}
		else if (e.getActionCommand().equals("zoom in")) {
			ResultsCommandManager.setImageZoom(zoom * 1.25);
//			imageMontagePanel.scaleImages(1.25);
//			zoomValue *= 1.25;
		}
		else if (e.getActionCommand().equals("zoom out")) {
			ResultsCommandManager.setImageZoom(zoom *= 0.8);
//			imageMontagePanel.scaleImages(0.8);
//			zoomValue *= 0.8;
		}
		else if (e.getActionCommand().equals("set view")) {
			int brainView = brainViewBox.getSelectedIndex();
			ResultsCommandManager.setBrainView(brainView);
		}
		else if (e.getActionCommand().equals("save brain slice images")) {
			doSaveAs();
		}
		else if (e.getActionCommand().equals("changeScheme")){
			
			ColorGradientScheme.setActiveScheme(
					(String) ((JComboBox)e.getSource()).getSelectedItem());
			reinitializePlot();
			revalidate();
			repaint();
		}
	}
	
	public void doSaveAs() {
		ImageWriter.saveBrainSlices(this);
	}
	
	public void updateImageRotation() {
		int rotation = mRepository.getImagePropertiesModel().getRotation();
		
		imageMontagePanel.rotateImages(rotation);
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {}

	///////////////////////////////////////////////////////////////////////////
	// Selection event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(SelectedDataTypeChangedEvent e) {
		updateSelection();
	}

	@Override
	public void notify(SelectedLvChangedEvent e) {
		updateSelection();
	}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {
		updateSelection();
	}

	@Override
	public void notify(SelectionEvent e) {
		updateSelection();
	}

	///////////////////////////////////////////////////////////////////////////
	// Brain image property event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BackgroundImageEvent e) {
		reinitializePlot();
		revalidate();
		repaint();
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
		updateUseDescriptionLabels();
	}

	@Override
	public void notify(UseLabelsEvent e) {
		updateUseLabels();
	}

	///////////////////////////////////////////////////////////////////////////
	// Filters event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(SliceFiltersEvent e) {
		updateSliceFilters();
	}

	@Override
	public void notify(ViewedLvsEvent e) {
		refreshPlot();
		revalidate();
		repaint();
	}

	///////////////////////////////////////////////////////////////////////////
	// Colour scale event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(ColourScaleEvent e) {
		updateColourScale();
		revalidate();
		repaint();
	}

	///////////////////////////////////////////////////////////////////////////
	// Data change event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(FlipVolumeEvent e) {
		reinitializePlot();
		revalidate();
		repaint();
	}

	@Override
	public void notify(LoadedVolumesEvent e) {
		refreshPlot();
	}
	
	@Override
	public void notify(InvertedLvEvent e) {
		refreshPlot();
		revalidate();
		repaint();
	}

	///////////////////////////////////////////////////////////////////////////
	// Single brain image viewer event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BrainViewEvent e) {
		setBrainView();
	}

	@Override
	public void notify(RotationEvent e) {
		updateImageRotation();
		revalidate();
		repaint();
	}

	@Override
	public void notify(ZoomEvent e) {
		// Obtain information about the current scroll state of the scrollpane
		JScrollBar horzBar = montageScroller.getHorizontalScrollBar();
		horzScrollPercent = (double)(horzBar.getValue() + (horzBar.getVisibleAmount() / 2.0) ) / 
				(double)(horzBar.getMaximum());// - horzBar.getVisibleAmount() );
		horzMax = horzBar.getMaximum();
		
		JScrollBar vertBar = montageScroller.getVerticalScrollBar();
		vertScrollPercent = (double)(vertBar.getValue() + (vertBar.getVisibleAmount() / 2.0) ) / 
				(double)(vertBar.getMaximum());// - vertBar.getVisibleAmount() );
		vertMax = vertBar.getMaximum();
		
		updateZoom();
		revalidate();
	}

	/*
	 * Ok, so an explanation of how this ends up working.  Basically, wait
	 * until both scroll bars have been resized (kept track of by horzDone
	 * and vertDone), then once both have been resized call the updateBars
	 * function.  If one of the bars is invisible, then it's ok to call
	 * update bars when only the one has been resized.  UpdateBars will set
	 * the value of both scrollbars, which causes this listener to trigger
	 * again.  zoomHack is used to prevent things from getting crazy.
	 */ 
	@Override
	public void adjustmentValueChanged(AdjustmentEvent event) {
		if (!event.getValueIsAdjusting() && zoomHack) {
			JScrollBar horzBar = montageScroller.getHorizontalScrollBar();
			JScrollBar vertBar = montageScroller.getVerticalScrollBar();
			
			if (event.getSource() == horzBar && horzMax != horzBar.getMaximum() &&
					horzScrollPercent != -1) {
				
				horzDone = true;
				
				if (vertDone || !vertBar.isVisible() ) {
					updateBars();
				}
			}
			else if (event.getSource() == vertBar && vertMax != vertBar.getMaximum() &&
				vertScrollPercent != -1) {
				
				vertDone = true;
				
				if (horzDone || !horzBar.isVisible() ) {
					updateBars();
				}
			}
		}
	}
	
	// Updates the values of both scroll bars.
	private void updateBars() {
		zoomHack = false;
		JScrollBar horzBar = montageScroller.getHorizontalScrollBar();
		JScrollBar vertBar = montageScroller.getVerticalScrollBar();
		
		int newVertValue = (int)(vertScrollPercent * (vertBar.getMaximum() ) ); // - vertBar.getVisibleAmount() ) );
		newVertValue -= vertBar.getVisibleAmount() / 2.0;
		
		vertBar.setValue(newVertValue);
		
		vertScrollPercent = -1;
		
		
		int newHorzValue = (int)(horzScrollPercent * (horzBar.getMaximum() ) ); // - horzBar.getVisibleAmount() ) );
		newHorzValue -= horzBar.getVisibleAmount() / 2.0;
		
		horzBar.setValue(newHorzValue);
		
		horzScrollPercent = -1;
		
		vertDone = false;
		horzDone = false;
		
		zoomHack = true;
	}

	@Override
	public void notify(BrainFilterEvent e) {
		imageMontagePanel.removeAllImages();
		imageMontagePanel.constructImages();
		revalidate();
	}

	public void notify(IncorrectLagsSelectedEvent e){};
}