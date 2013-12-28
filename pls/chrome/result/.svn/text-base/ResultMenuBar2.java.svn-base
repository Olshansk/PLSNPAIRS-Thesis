package pls.chrome.result;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import npairs.io.NiftiIO;
import pls.chrome.result.blvplot.BrainLatentVariablesPlot;
import pls.chrome.result.blvplot.PlotTypeTabbedPane;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;
import pls.chrome.result.controller.observer.brainimageproperties.BackgroundImageEvent;
import pls.chrome.result.controller.observer.brainimageproperties.BrainImagePropertiesAdaptor;
import pls.chrome.result.controller.observer.datachange.DataChangeAdaptor;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionAdaptor;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.view.AbstractPlot;
import pls.chrome.result.view.AttachDetachOption;
import pls.chrome.result.view.RegressionPlot;
import pls.chrome.result.view.scatterplot.EnhancedScatterPlot;
import pls.sessionprofile.NiftiAnalyzeImage;
import pls.shared.AnalyzeImageFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import pls.shared.NiftiImageFileFilter;

@SuppressWarnings("serial")
public class ResultMenuBar2 extends JMenuBar implements ActionListener, Observer{
	
	private JMenuItem mRemoveBgImage;
	
	private JMenuItem mUndo;
	private JMenuItem mRedo;
	
	private JMenuItem mVoxelExtraction;
	
	private JMenuItem mShowHideLeftPanel;
	
	private GeneralRepository mRepository;
	
	private DetachableTabbedPane mTabs;
	private PlotTypeTabbedPane mplotTabs;
	private AttachDetachOption mainView;
	
	// Used to remember the location of the last file opened.
	private String bgImagePath = null;
	
	//Anatomical image browser
	private JFileChooser chooser;
	
	public ResultMenuBar2(GeneralRepository repository, 
						DetachableTabbedPane tabs) {
		
		mTabs = tabs;
		mainView = repository.getPlotManager().getPlots().get(0);
		mRepository = repository;
		mRepository.getPublisher().registerObserver(this);
		
		/*Check that the main brain viewer is available and if it is grab
		the tabbed pane that holds the scatter plot. This code exists for
		debug purposes because sometimes I disable the main brain viewer's
		creation*/ 
		
		if(mainView instanceof BrainLatentVariablesPlot){
			mplotTabs = ((BrainLatentVariablesPlot) mainView).getPlotTabs();
		}else{
			mplotTabs = null;
		}
		///////////////////////////////////////////////////////////////////////
		// File Menu
		///////////////////////////////////////////////////////////////////////
		JMenu fileMenu = new JMenu("File");
		JMenuItem loadBgImage = new JMenuItem("Load Background Image",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		mRemoveBgImage = new JMenuItem("Remove Background Image",
			new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Remove16.gif")));
		
		final JMenuItem savePlot = new JMenuItem("Save Brain Slice Images As ...",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Save16.gif")));
		final JMenuItem saveScatterPlot = new JMenuItem("Save Scatter plot As ...",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Save16.gif")));
		
		JMenuItem addRemoveResultFiles = new JMenuItem("Add/Remove Result Files",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		JMenuItem exit = new JMenuItem("Exit",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Stop16.gif")));
		
		fileMenu.add(loadBgImage);
		fileMenu.add(mRemoveBgImage);
		fileMenu.addSeparator();
		fileMenu.add(savePlot);
		fileMenu.add(saveScatterPlot);
		fileMenu.add(addRemoveResultFiles);
		fileMenu.addSeparator();
		fileMenu.add(exit);
		
		add(fileMenu);
		
		loadBgImage.setActionCommand("load bg");
		loadBgImage.addActionListener(this);
		mRemoveBgImage.setEnabled(false);
		mRemoveBgImage.setActionCommand("remove bg");
		mRemoveBgImage.addActionListener(this);
		savePlot.setActionCommand("save plot");
		savePlot.addActionListener(this);
		saveScatterPlot.setEnabled(false);
		saveScatterPlot.setActionCommand("save scatterplot");
		saveScatterPlot.addActionListener(this);
		addRemoveResultFiles.setActionCommand("add/remove files");
		addRemoveResultFiles.addActionListener(this);
		exit.setActionCommand("exit");
		exit.addActionListener(this);
		
		BrainImagePropertiesAdaptor removeButtonWatcher = createWatcher();
		mRepository.getPublisher().registerObserver(removeButtonWatcher);
		///////////////////////////////////////////////////////////////////////
		// Edit Menu
		///////////////////////////////////////////////////////////////////////
		JMenu editMenu = new JMenu("Edit");
		mUndo = new JMenuItem("Undo",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Undo16.gif")));
		mUndo.setEnabled(false);
		mRedo = new JMenuItem("Redo",
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Redo16.gif")));
		mRedo.setEnabled(false);
		
		editMenu.add(mUndo);
		editMenu.add(mRedo);
		
		add(editMenu);
		
		mUndo.setActionCommand("undo");
		mUndo.addActionListener(this);
		mRedo.setActionCommand("redo");
		mRedo.addActionListener(this);
		
		///////////////////////////////////////////////////////////////////////
		// Options Menu
		///////////////////////////////////////////////////////////////////////
		JMenu optionsMenu = new JMenu("Options");
		mShowHideLeftPanel = new JMenuItem("Hide Left Panel in Main Viewer");
		JMenuItem flipVolumes = new JMenuItem("Flip Volumes");
		flipVolumes.setEnabled(false);
		mVoxelExtraction = new JMenuItem("Multiple Voxels Extraction");
		JMenuItem selectLibrary = new JMenuItem("Select imaging library");
		
		optionsMenu.add(mShowHideLeftPanel);
		optionsMenu.add(flipVolumes);
		optionsMenu.add(mVoxelExtraction);
		optionsMenu.add(selectLibrary);
		
		add(optionsMenu);
		
		mShowHideLeftPanel.setActionCommand("show/hide left panel");
		mShowHideLeftPanel.addActionListener(this);
		flipVolumes.setActionCommand("flip volumes");
		flipVolumes.addActionListener(this);
		flipVolumes.setEnabled(false);
		mVoxelExtraction.setActionCommand("multiple voxels extraction");
		mVoxelExtraction.addActionListener(this);
		selectLibrary.setActionCommand("select library");
		selectLibrary.addActionListener(this);
		
		DataChangeAdaptor d = new DataChangeAdaptor(){
			@Override
			public void notify(LoadedVolumesEvent e) {
				refreshMultipleVoxelsButton();
			}
		};
		
		SelectionAdaptor s = new SelectionAdaptor(){
			@Override
			public void notify(SelectedVolumeChangedEvent e) {
				refreshMultipleVoxelsButton();
			}
		};
		
		mRepository.getPublisher().registerObserver(d);
		mRepository.getPublisher().registerObserver(s);
		if(!(mRepository.getGeneral() instanceof PlsResultModel)){
			mVoxelExtraction.setEnabled(false);
		}
		///////////////////////////////////////////////////////////////////////
		// Help Menu
		///////////////////////////////////////////////////////////////////////
		JMenu helpMenu = new JMenu("Help");
		JMenuItem helpContents = new JMenuItem("Help Contents",
				new ImageIcon(this.getClass().getResource(
				"/toolbarButtonGraphics/general/Help16.gif")));

		helpContents.addActionListener(
				new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						JOptionPane.showMessageDialog(null, "This is the " +
								"result viewer which is used" +
								" to view the contents of result files.",
								"Help Contents",
								JOptionPane.INFORMATION_MESSAGE);
					}
				}
		);

		JMenuItem about = new JMenuItem("About PLS and NPAIRS", new ImageIcon(this.getClass().
        		getResource("/toolbarButtonGraphics/general/About16.gif")));
        about.setMnemonic('A');
        about.getAccessibleContext().setAccessibleDescription("More about this program");
        about.addActionListener(
        		new ActionListener() {
					public void actionPerformed(ActionEvent e) {
        				JOptionPane.showMessageDialog(null, "This is a Java translation and enhancement of PLS" +
        						" (originally written in Matlab) \nand NPAIRS (originally written in IDL)." +
        						"\nVersion " + GlobalVariablesFunctions.getVersion() + ".\nPlease contact Stephen Strother " +
        						"<sstrother@rotman-baycrest.on.ca> \nor Anita Oder <aoder@rotman-baycrest.on.ca> " +
        						"to report any bugs or make comments, \nor visit our open-source project at: " +
        						"<http://code.google.com/p/plsnpairs/>.", "About PLS and NPAIRS", JOptionPane.INFORMATION_MESSAGE);
        				}
        			}
        		);
        helpMenu.add(helpContents);
		helpMenu.add(about);
		
		add(helpMenu);
		
		tabs.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				JTabbedPane tabs = (JTabbedPane) e.getSource();
				if (tabs.getSelectedIndex() == 0) {
					savePlot.setText("Save Brain Slice Images As ...");
					mShowHideLeftPanel.setEnabled(true);
				} else {
					savePlot.setText("Save Plot As ...");
					mShowHideLeftPanel.setEnabled(false);
				}
			}
		});
		
		if(mplotTabs != null){
			mplotTabs.addChangeListener(new ChangeListener(){
				public void stateChanged(ChangeEvent e){
					JTabbedPane plotTabs = (JTabbedPane) e.getSource();
					if (plotTabs.getSelectedComponent() 
										instanceof EnhancedScatterPlot){
						
						EnhancedScatterPlot plot = (EnhancedScatterPlot)
							plotTabs.getSelectedComponent();
						
						if(plot.okToSave()){
							saveScatterPlot.setEnabled(true);
						}
					}else{
						saveScatterPlot.setEnabled(false);
					}
					
				}
			});
		}
	}

	/**
	 * Creates a new observer so that when a new anatomical image
	 * is loaded/removed the "Remove Background Image" button is
	 * enabled/disabled accordingly. This may seem like over kill but it 
	 * is necessary because images may be unloaded/loaded through undo/redo
	 * commands that have no knowledge of this class unless there exists
	 * an observer.
	 * @return
	 */
	private BrainImagePropertiesAdaptor createWatcher() {
		BrainImagePropertiesAdaptor bipa = new BrainImagePropertiesAdaptor(){
			@Override
			public void notify(BackgroundImageEvent e) {
				double[] bgdata = mRepository.getGeneral().getBgImageData();
				if(bgdata == null){
					mRemoveBgImage.setEnabled(false);
				}else{
					mRemoveBgImage.setEnabled(true);
				}
			}
		};
		return bipa;
	}

	/**
	 * Event handling function.  Performs correct actions when menu items
	 * have been clicked.
	 */
	@Override
	public void actionPerformed(ActionEvent event) {
		String actionCommand = event.getActionCommand();
		
		////////////////////
		// File Menu
		////////////////////
		// Load a background image
		if (actionCommand.equals("load bg") ) {
			loadBackgroundImage();
		}
		// Remove a background image
		else if (actionCommand.equals("remove bg") ) {
			removeBackgroundImage();
		}
		// Save the current plot
		else if (actionCommand.equals("save plot") ) {
			savePlotAs();
		}
		else if (actionCommand.equals("save scatterplot")){
			saveScatterPlotAs();
		}
		// Bring up the add/remove files dialog
		else if (actionCommand.equals("add/remove files") ) {
			LoadedVolumesDialog.showDialog(this, mRepository);
		}
		// Exit the results displayer
		else if (actionCommand.equals("exit") ) {
			JOptionPane.getFrameForComponent(this).dispose();
		}
		//////////////////
		// Edit Menu
		//////////////////
		// Undo the last command
		else if (actionCommand.equals("undo") ) {
			ResultsCommandManager.undo();
		}
		// Redo the last command
		else if (actionCommand.equals("redo") ) {
			ResultsCommandManager.redo();
		}
		//////////////////
		// Options Menu
		//////////////////
		// hide the left panel visible from the main brain viewer
		else if (actionCommand.equals("show/hide left panel") ) {
			if (mShowHideLeftPanel.getText().equals("Hide Left Panel in Main Viewer")) {
				ResultsCommandManager.setLeftPanelVisible(false);
				mShowHideLeftPanel.setText("Show Left Panel in Main Viewer");
			} else {
				ResultsCommandManager.setLeftPanelVisible(true);
				mShowHideLeftPanel.setText("Hide Left Panel in Main Viewer");
			}
		}
		// invoke the flip volumes dialog
		else if (actionCommand.equals("flip volumes") ) {
			VolumeOrientationDialog.showDialog(this, mRepository);
		}
		// invoke the multiple voxels extraction dialog
		else if (actionCommand.equals("multiple voxels extraction") ) {
			invokeMultipleVoxelsExtraction();
		}
		else if(actionCommand.equals("select library")){
			changeAnatomicalLibrary();
		}
	}

	/**
	 * Temporary fix for issue 
	 * http://code.google.com/p/plsnpairs/issues/detail?id=78
	 * Change the library used to render anatomical images.
	 */
	private void changeAnatomicalLibrary() {
		final JFrame lSelect = new JFrame();
		Container cpane = lSelect.getContentPane();
		final JComboBox lSelector = new JComboBox();
		final JButton ok = new JButton("Ok");
		final JButton cancel = new JButton("Cancel");
		final boolean niftiInUse = mRepository.getAnatomicalLib();
		
		//The lib that is added first is the currently selected lib.
		if(niftiInUse == true){
			lSelector.addItem("Nifti1Dataset library");
			lSelector.addItem("MindSeer library");
		}else{
			lSelector.addItem("MindSeer library");
			lSelector.addItem("Nifti1Dataset library");
		}
		
		ActionListener al = new ActionListener(){

			@Override
			public void actionPerformed(ActionEvent e) {
				boolean lib;
				
				if(e.getSource() == ok){
					
					if(lSelector.getSelectedItem().equals("MindSeer library")){
						lib = false;
					}
					else{
						lib = true;
					}
					
					//Load the images again due to the library change
					if(niftiInUse != lib){
						if(changeLibBackgroundImage(lib)){
							//Update the repository with the new lib selection.
							mRepository.setAnatomicalLib(lib);
						}
					}
					lSelect.dispose();
				}
				else{
					lSelect.dispose();
				}
			}
		};

		ok.addActionListener(al);
		cancel.addActionListener(al);
		
		cpane.setLayout(new FlowLayout());
		cpane.add(lSelector);
		cpane.add(ok);
		cpane.add(cancel);
		lSelect.setTitle("Select imaging library");
		lSelect.pack();
		lSelect.setVisible(true);
		lSelect.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	/**
	 * Save the currently selected plot as an image file.
	 */
	private void savePlotAs() {
		Component tab = mTabs.getSelectedComponent();
		if (tab instanceof AbstractPlot) {
			AbstractPlot plot = (AbstractPlot) tab;
			plot.doSaveAs();
		} else if (tab instanceof BrainLatentVariablesPlot) {

			ResultsCommandManager.saveBrainSliceImages();
						
		}else if (tab instanceof RegressionPlot){
			RegressionPlot plot = (RegressionPlot) tab;
			plot.doSaveAs();									
		}
	}
	
	/**
	 * Save the scatter plot as an image file
	 */
	private void saveScatterPlotAs(){
		EnhancedScatterPlot plot = (EnhancedScatterPlot) 
		mplotTabs.getSelectedComponent();
		
		plot.doSaveAs();
	}
	
	private void invokeMultipleVoxelsExtraction() {
		
		// First, checks if any datamat files are associated with the currently-selected
		// result file before moving on.
		ResultModel m = mRepository.getGeneral();
		if(!(m instanceof PlsResultModel)) return;
		PlsResultModel model = (PlsResultModel) m;
		ArrayList<ArrayList<String>> datamatProfiles = model.getDatamatProfiles();
		if (datamatProfiles == null || datamatProfiles.isEmpty()) {
			GlobalVariablesFunctions.showErrorMessage("The currently-selected volume, " +
					model.getFilename() + ", does not contain any datamat files");
			return;
		}
		
		Frame frame = JOptionPane.getFrameForComponent(this);
		
		MultipleVoxelExtraction extraction = new MultipleVoxelExtraction(frame, "", model);
		if (!extraction.isVoxelLocationFileLoaded()) {
			return;
		}
		
		extraction.saveFiles();
	}
	
	/**
	 * Sets the background image to null via a BackgroundImageCommand.
	 */
	private void removeBackgroundImage() {
		ResultsCommandManager.setBackgroundImage(null);
	}
	
	/**
	 * Change the library used to view the current anatomical background
	 * image by reloading it using the currently selected library
	 * @param lib the library to use, true if nifti, false if mindseer.
	 * @return true if the image displayed correctly using the selected lib.
	 */
	private boolean changeLibBackgroundImage(boolean lib){
		File imagePath = mRepository.getBgImagePath();
		double[] bgImageData;
		
		if(imagePath == null) return true; //no image loaded yet.
		
		//If using mind seer imaging library...
		if(!lib){
			NiftiAnalyzeImage bgImage;
			try{
				bgImage = new NiftiAnalyzeImage(imagePath.getParent(),
												imagePath.getName());
			}catch(Exception e){
				GlobalVariablesFunctions.showErrorMessage("Background image file "
						+ imagePath.getAbsolutePath() + " could not be loaded.");
				return false;
			}
			bgImageData = getBgImageData(null,bgImage,false);
		}
		else{ //nifti1 lib.
			bgImageData = getBgImageData(bgImagePath,null,true);
		}
				
		if(bgImageData != null){
			Set<String> files = mRepository.getModels();
			for(String model : files){
				mRepository.getGeneral(model).setBgImageData(bgImageData);
			}
			mRepository.getPublisher().publishEvent(new BackgroundImageEvent());
		}
		
		return true;
	}
	
	/**
	 * Loads a background image as a NiftiAnalyzeImage, and executes a
	 * BackgroundImageCommand.
	 */
	private void loadBackgroundImage() {
		if(chooser == null){
			chooser = new JFileChooser(".");
			chooser.addChoosableFileFilter(new AnalyzeImageFileFilter());
			chooser.addChoosableFileFilter(new NiftiImageFileFilter());
			chooser.setPreferredSize(new Dimension(640,480));
		}
		
		
		int option = chooser.showDialog(this, "Load Background Image");
		
		if (option == JFileChooser.APPROVE_OPTION) {
			bgImagePath = chooser.getSelectedFile().getAbsolutePath();
			ResultsCommandManager.setBackgroundImage(
					chooser.getSelectedFile());
		}
	}

	/**
	 * Function to get the data in a NiftiAnalyzeImage as an array
	 * of doubles.  Also verifies that the dimensions of the NiftiImage
	 * are the same as the currently loaded models.
	 */

	private double[] getBgImageData(String path, NiftiAnalyzeImage bgImage,
			boolean useNifti) {
		ResultModel model = mRepository.getGeneral();
		int[] stDims = model.getBrainData().getDimensions();
		
		int[] bgDims = null;
		double[] bgImageData = null;
		
		if(useNifti){
			try {
				bgDims = NiftiIO.getVolDims3D(path);
				bgImageData = NiftiIO.readNiftiData(path, 0);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}else{ //mindseer
			bgDims = bgImage.get4DDimensions();
			bgImageData = bgImage.getData();
		}
		

		// Compares the resolutions of the background image and brain image.
		// Note: For the background image, its resolution is stored as [x y z t]
		// while the brain image's resolution is stored as [x y t z].
//		int[] bgDims = bgImage.get4DDimensions();
		if (bgDims[0] != stDims[0] || bgDims[1] != stDims[1]
				|| bgDims[2] != stDims[3]) {
			GlobalVariablesFunctions.showErrorMessage("Background image file "
					+ " does not have a " +
					"matching resolution to the brain image.");
			return null;
		}

//		double[] bgImageData = bgImage.getData();
		bgImageData = MLFuncs.normalize(bgImageData);
		
		return bgImageData;
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event response functions
	///////////////////////////////////////////////////////////////////////////
	/**
	 * Refreshes the undo and redo menu items with the command strings
	 * from the CommandManager's command stack.
	 */
	private void refreshUndoRedo() {
		if (ResultsCommandManager.isUndoAvailable() ) {
			mUndo.setText("Undo " + ResultsCommandManager.getNextUndo().getCommandLabel() );
			mUndo.setEnabled(true);
		}
		else {
			mUndo.setText("Undo");
			mUndo.setEnabled(false);
		}
		
		if (ResultsCommandManager.isRedoAvailable() ) {
			mRedo.setText("Redo " + ResultsCommandManager.getNextRedo().getCommandLabel() );
			mRedo.setEnabled(true);
		}
		else {
			mRedo.setText("Redo");
			mRedo.setEnabled(false);
		}
	}
	
	/**
	 * Checks if the currently selected model is a PlsModel.  Multiple
	 * voxels extraction is only enabled for PlsModels.
	 */
	private void refreshMultipleVoxelsButton() {
		mVoxelExtraction.setEnabled(
				mRepository.getGeneral() instanceof PlsResultModel);
	}
	
	
	@Override
	public void finalize() throws Throwable {
		
		super.finalize();
	}
	
	@Override
	public void notify(Event e) {
		refreshUndoRedo();
	}
	
}
