package pls.chrome.result;

import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;

import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.blvplot.BrainLatentVariablesPlot;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.view.AbstractPlot;
import pls.chrome.result.view.ScatterPlot;
import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.ProgressDialog;
import pls.sessionprofile.NiftiAnalyzeImage;
import pls.shared.AnalyzeImageFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NiftiImageFileFilter;

/**
 * An extension of the BaseSaveMenuBar.  It adds more items to the file
 * menu so that an anatomical image can be underlayed in the background of
 * the brain latent variables plot in the results viewer.
 * Update: Wed 29th, Sep 2010. This class is no longer in use? -Fletcher
 */
@SuppressWarnings("serial")
final class ResultMenuBar extends BaseSaveMenuBar implements ActionListener,
DataChangeObserver, SelectionObserver {
	private ResultFrame frame = null;
	private String fileType = null;
	private DetachableTabbedPane tabs;
	private AbstractResultsDisplayer worker = null;
	private NiftiAnalyzeImage bgImage = null;
	private String bgImageDataPath = ".";
	private String bgImageFileName = null;

	private JMenuItem removeBackgroundImage;
	private JMenuItem multipleVoxelsExtraction;
	private final JMenuItem showHideLeftPanel;
	
	private GeneralRepository repository;
	private JMenuItem undo;
	private JMenuItem redo;

	public ResultMenuBar(ResultFrame frame, String fileType, DetachableTabbedPane tabs,
			FileFilter fileFilter, String fileExtension) {
		super(frame);
		setFileFilter(fileFilter, fileExtension);
		this.tabs = tabs;
		this.frame = frame;
		this.fileType = fileType;

		JMenu fileMenu = getMenu(0);
		
		// Remove the "load" option from the file menu since it is not
		// needed for the results displayer.
		fileMenu.remove(0);
		
		// Disables the "save" and "save as" options from the file menu
		// since it has not yet been determined what will be saved.
		fileMenu.getItem(0).setEnabled(false);
		fileMenu.getItem(1).setEnabled(false);

		// File menu's options
		JMenuItem loadBackgroundImage = new JMenuItem("Load Background Image",
				new ImageIcon(this.getClass().getResource(
						"/toolbarButtonGraphics/general/Open16.gif")));
		loadBackgroundImage.setMnemonic('B');
		loadBackgroundImage.getAccessibleContext().setAccessibleDescription(
				"Load background image from file");
		loadBackgroundImage.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_loadBackgroundImage();
			}
		});
		fileMenu.add(loadBackgroundImage, 0);

		removeBackgroundImage = new JMenuItem("Remove Background Image",
				new ImageIcon(this.getClass().getResource(
						"/toolbarButtonGraphics/general/Delete16.gif")));
		removeBackgroundImage.setMnemonic('R');
		removeBackgroundImage.getAccessibleContext().setAccessibleDescription(
				"Remove background image");
		removeBackgroundImage.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_removeBackgroundImage();
			}
		});
		removeBackgroundImage.setEnabled(false);
		fileMenu.add(removeBackgroundImage, 1);

		fileMenu.insertSeparator(2);
		
		final JMenuItem savePlotAs = new JMenuItem("Save Brain Slice Images As ...", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/SaveAs16.gif")));
		savePlotAs.getAccessibleContext().setAccessibleDescription("Save Plot As ...");
		savePlotAs.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_savePlotAs();
			}
		});
		
		fileMenu.add(savePlotAs, 3);

		fileMenu.insertSeparator(4);
		
		JMenuItem manageVolumes = new JMenuItem("Add/Remove Result Files", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		manageVolumes.getAccessibleContext().setAccessibleDescription("Add/Remove Result Files");
		manageVolumes.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_invokeVolumesDialog();
			}
		});
		fileMenu.add(manageVolumes, 5);
		
		// Edit Menu
		JMenu editMenu = new JMenu("Edit");
		editMenu.setMnemonic('E');
		editMenu.getAccessibleContext().setAccessibleDescription("Edit menu");
		
		undo = new JMenuItem("Undo");
		undo.setActionCommand("undo");
		undo.addActionListener(this);
		editMenu.add(undo);
		
		redo = new JMenuItem("Redo");
		redo.setActionCommand("redo");
		redo.addActionListener(this);
		editMenu.add(redo);
		
		refreshUndoRedo();

		add(editMenu, 1);
		
		// Options Menu
		JMenu optionsMenu = new JMenu("Options");
		optionsMenu.setMnemonic('O');
        optionsMenu.getAccessibleContext().setAccessibleDescription("Options menu");

		showHideLeftPanel = new JMenuItem("Hide Left Panel in Main Viewer");
		optionsMenu.add(showHideLeftPanel);
		showHideLeftPanel.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_showHideLeftPanel();
			}
		});
		
		JMenuItem flipVolumes = new JMenuItem("Flip Volumes");
		optionsMenu.add(flipVolumes);
		flipVolumes.setEnabled(false);
		flipVolumes.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_invokeFlipDialog();
			}
		});

		multipleVoxelsExtraction = new JMenuItem("Multiple Voxels Extraction");
		optionsMenu.add(multipleVoxelsExtraction);
		multipleVoxelsExtraction.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_invokeMultipleVoxelsExtraction();
			}
		});
		
		if (fileType.equals(GlobalVariablesFunctions.NPAIRS)) {
			multipleVoxelsExtraction.setEnabled(false);
		}
		
		add(optionsMenu, 2);
		
		tabs.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				JTabbedPane tabs = (JTabbedPane) e.getSource();
				if (tabs.getSelectedIndex() == 0) {
					savePlotAs.setText("Save Brain Slice Images As ...");
					showHideLeftPanel.setEnabled(true);
				} else {
					savePlotAs.setText("Save Plot As ...");
					showHideLeftPanel.setEnabled(false);
				}
			}
		});
	}

	protected void _invokeFlipDialog() {
		VolumeOrientationDialog.showDialog(this, repository);
	}

	private void _savePlotAs() {
		Component tab = tabs.getSelectedComponent();
		if (tab instanceof AbstractPlot) {
			AbstractPlot plot = (AbstractPlot) tab;
			plot.doSaveAs();
		} else if (tab instanceof BrainLatentVariablesPlot) {
			ResultsCommandManager.saveBrainSliceImages();
		} else if (tab instanceof ScatterPlot) {
			ScatterPlot plot = (ScatterPlot) tab;
			plot.doSaveAs();
		}
	}

	@Override
	/**
	 * This function is not called because the load button has been removed 
	 * from the result menu bar.
	 */
	public void load() {
		tabs.removeAll();

		if (fileType.equals(GlobalVariablesFunctions.NPAIRS)) {
			worker = new NpairsResultsDisplayer(tabs, fileName);
		} else {
			worker = new PlsResultsDisplayer(tabs, fileName);
		}
		final ProgressDialog dialog = new ProgressDialog(frame, 16, worker);
		
		// Depending on the operating system, this line may be reached
		// before the progress dialog has finished, meaning the results
		// displayer has not been completely loaded yet. We do not want this.
		repository = worker.getRepository();
		
		final ResultMenuBar bar = this;
		// If repository is null, then it has not been initialized by worker
		// yet. This means that the results displayer has not been loaded yet,
		// so we have to wait.
		if (repository == null) {
			
			new Thread() {
				public void run() {
				
					// Waits until the progress dialog has finished,
					// meaning the results displayer has been completely loaded.
					// Then we can register the observer.
					while (!dialog.isComplete());
					repository = worker.getRepository();
					repository.getPublisher().registerObserver(bar);
				}
			}.start();
		} else {
			repository.getPublisher().registerObserver(this);
		}
	}

	private void _loadBackgroundImage() {
		JFileChooser chooser = new JFileChooser(bgImageDataPath);
		chooser.addChoosableFileFilter(new AnalyzeImageFileFilter());
		chooser.addChoosableFileFilter(new NiftiImageFileFilter());
		
		if (bgImageFileName != null) {
			chooser.setSelectedFile(new File(bgImageDataPath, bgImageFileName));
		}

		int option = chooser.showDialog(ResultMenuBar.this, "Load Background Image");
		if (option == JFileChooser.APPROVE_OPTION) {
			bgImageDataPath = chooser.getSelectedFile().getParent();
			bgImageFileName = chooser.getSelectedFile().getName();
			try {
				bgImage = new NiftiAnalyzeImage(bgImageDataPath, bgImageFileName);
				boolean result = setBackgroundImage(bgImage, bgImageFileName);
				removeBackgroundImage.setEnabled(result);
			} catch (Exception e) {
				GlobalVariablesFunctions.showErrorMessage("Background image file "
						+ bgImageFileName + " could not be loaded.");
			}
		}
	}
	
	private boolean setBackgroundImage(NiftiAnalyzeImage bgImage, String fileName) {
		ResultModel model = repository.getGeneral();
		String resultFile = model.getFilename();
		int[] stDims = model.getBrainData().getDimensions();

		// Compares the resolution of the loaded background image to the brain
		// image.
		// If they are different, then the background image is not used.
		// Note: For the background image, its resolution is stored as [x y z t]
		// while
		// the brain image's resolution is stored as [x y t z].
		int[] bgDims = bgImage.get4DDimensions();
		if (bgDims[0] != stDims[0] || bgDims[1] != stDims[1]
				|| bgDims[2] != stDims[3]) {
			GlobalVariablesFunctions.showErrorMessage("Background image file "
													+ fileName
													+ " does not have a matching resolution to the brain image.");
			return false;
		}

		double[] bgImageData = bgImage.getData();
		bgImageData = MLFuncs.normalize(bgImageData);

		// Set the current bg image.
		model.setBgImageData(bgImageData);
		//ResultsCommandManager.reInitializeImages(resultFile);
		
		return true;
	}
	
	public void actionPerformed(ActionEvent evt) {
		if (evt.getActionCommand().equals("undo")) {
			ResultsCommandManager.undo();
		}
		else if (evt.getActionCommand().equals("redo")) {
			ResultsCommandManager.redo();
		}
	}

	private void _removeBackgroundImage() {
		try {
			ResultModel model = repository.getGeneral();
			String resultFile = model.getFilename();
			
			// Sets the current bg image to null.
			model.setBgImageData(null);
			//ResultsCommandManager.reInitializeImages(resultFile);

			removeBackgroundImage.setEnabled(false);
		} catch (Exception e) {
			GlobalVariablesFunctions.showErrorMessage("Background image file could not be removed.");
		}
	}
	
	private void _invokeVolumesDialog() {
		LoadedVolumesDialog.showDialog(frame, repository);
	}
	
	private void _showHideLeftPanel() {
		if (showHideLeftPanel.getText().equals("Hide Left Panel in Main Viewer")) {
			ResultsCommandManager.setLeftPanelVisible(false);
			showHideLeftPanel.setText("Show Left Panel in Main Viewer");
		} else {
			ResultsCommandManager.setLeftPanelVisible(true);
			showHideLeftPanel.setText("Hide Left Panel in Main Viewer");
		}
	}
	
	private void _invokeMultipleVoxelsExtraction() {
		
		// First, checks if any datamat files are associated with the currently-selected
		// result file before moving on.
		PlsResultModel model = (PlsResultModel) repository.getGeneral();
		ArrayList<ArrayList<String>> datamatProfiles = model.getDatamatProfiles();
		if (datamatProfiles == null || datamatProfiles.isEmpty()) {
			GlobalVariablesFunctions.showErrorMessage("The currently-selected volume, " +
					model.getFilename() + ", does not contain any datamat files");
			return;
		}
		
		MultipleVoxelExtraction extraction = new MultipleVoxelExtraction(frame, filePath, model);
		if (!extraction.isVoxelLocationFileLoaded()) {
			return;
		}
		
		extraction.saveFiles();
	}
	
	public void refreshUndoRedo() {
		if (ResultsCommandManager.isUndoAvailable() ) {
			undo.setText("Undo " + ResultsCommandManager.getNextUndo().getCommandLabel() );
			undo.setEnabled(true);
		}
		else {
			undo.setText("Undo");
			undo.setEnabled(false);
		}
		
		if (ResultsCommandManager.isRedoAvailable() ) {
			redo.setText("Redo " + ResultsCommandManager.getNextRedo().getCommandLabel() );
			redo.setEnabled(true);
		}
		else {
			redo.setText("Redo");
			redo.setEnabled(false);
		}
	}
	
	public void finalize() throws Throwable {
		System.out.println("ResultMenuBar killed.");
		
		super.finalize();
	}
	
	private void enableMultipleVoxelsExtraction() {
		multipleVoxelsExtraction.setEnabled(repository.getGeneral() instanceof PlsResultModel);
	}
	
	private void enableRemoveBackgroundImage() {
		removeBackgroundImage.setEnabled(repository.getGeneral().getBgImageData() != null);
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {
		refreshUndoRedo();
	}

	///////////////////////////////////////////////////////////////////////////
	// Data change event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {
		enableMultipleVoxelsExtraction();
		enableRemoveBackgroundImage();
	}
	
	@Override
	public void notify(InvertedLvEvent e) {}

	///////////////////////////////////////////////////////////////////////////
	// Selection event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(SelectedDataTypeChangedEvent e) {}

	@Override
	public void notify(SelectedLvChangedEvent e) {}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {
		enableMultipleVoxelsExtraction();
		enableRemoveBackgroundImage();
	}

	@Override
	public void notify(SelectionEvent e) {}
}