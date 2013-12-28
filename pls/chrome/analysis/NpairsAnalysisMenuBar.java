package pls.chrome.analysis;

import java.util.ArrayList;
import java.util.Vector;

import npairs.io.NpairsjIO;

import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.BaseMenuBar;
import pls.shared.NpairsAnalysisSetupERFileFilter;
import pls.shared.NpairsAnalysisSetupFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NpairsfMRIResultFileFilter;

import javax.swing.ImageIcon;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JCheckBox;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.table.DefaultTableModel;

import nl.jj.swingx.gui.modal.JModalFrame;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.Dimension;
import java.awt.Toolkit;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLStructure;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;

import extern.NewMatFileReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;


@SuppressWarnings("serial")
public class NpairsAnalysisMenuBar extends BaseSaveMenuBar {

	public boolean useCondsAsClasses = true;

	//	private boolean debug = false;

	private NpairsAnalysisFrame npairsFrame;
	//private JFrame parent;

	//	private Vector<Integer> classSelection;
	private String[] classNames;
	private JMenuItem deselectCVAClasses;

	/* For Junit testing only. Notifies the testing program that an error 
	 * occured when attempting to load a analysis setup file.
	 */
	private boolean loadErr = false; 

	private JCheckBoxMenuItem saveLotsOfFiles;
	private JCheckBoxMenuItem saveSplitFiles;
	private JCheckBoxMenuItem rotatePCsToImgSpace;
	private JCheckBoxMenuItem saveDataPostPCA;
	private JCheckBoxMenuItem useMatlabForInitEVD;
	private JCheckBoxMenuItem calcProcWithEigims;

	public NpairsAnalysisMenuBar(NpairsAnalysisFrame npairsFrame, 
			Vector<Integer> classSelection,
			String[] classNames, boolean isBlocked) {
		super(npairsFrame);

		//		this.classSelection = classSelection;
		this.npairsFrame = npairsFrame;
		this.classNames = classNames;
		buildDeselectMenu(npairsFrame, classSelection, classNames);
		buildOptionsMenu(npairsFrame);
		// Retrieve the file menu to modify its options.
		JMenu fileMenu = getMenu(0);

		// Add a new menu item for clearing all the fields in the analysis frame.
		JMenuItem clear = new JMenuItem("Clear",
				new ImageIcon(this.getClass().getResource(
				"/toolbarButtonGraphics/general/Delete16.gif")));
		clear.setMnemonic('C');
		clear.getAccessibleContext().setAccessibleDescription("Clear");
		clear.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				clearFields();
			}
		});
		fileMenu.add(clear, 3);
		fileMenu.insertSeparator(4);

		if (isBlocked) {
			setFileFilter(new NpairsAnalysisSetupFileFilter(), "_NPAIRSAnalysisSetup.mat");
		}
		else {
			setFileFilter(new NpairsAnalysisSetupERFileFilter(), "_NPAIRSAnalysisSetupER.mat");
		}
	}

	private void buildDeselectMenu(NpairsAnalysisFrame parent, 
			Vector<Integer> classSelection, 
			String[] classNames) {
		JMenu deselectMenu = new JMenu("Deselect");
		add(deselectMenu, 1);
		deselectCVAClasses = new JMenuItem("Deselect CVA classes");
		deselectCVAClasses.getAccessibleContext().setAccessibleDescription(
		"Filter out CVA classes you don't want to be a part of the analysis");
		deselectCVAClasses.addActionListener(new DeselectClassListener(parent, classSelection, classNames));

		deselectMenu.add(deselectCVAClasses);
	}

	private void buildOptionsMenu(NpairsAnalysisFrame parent) {

		final NpairsAnalysisFrame npairsFrame = parent;
		final JMenu optionMenu = new JMenu("Options");
		add(optionMenu, 2);

		saveLotsOfFiles = new JCheckBoxMenuItem("Save lots of output files");
		saveLotsOfFiles.setToolTipText("<html>If checked, full-data and summary results will be " 
				+ "saved in individual files.  <br> Some other output files, e.g. mask volume indices, "
				+ "will also be saved. <br> All files saved when this box is not checked will still be "
				+ "saved.");
		saveLotsOfFiles.setSelected(false);	// don't save by default

		saveLotsOfFiles.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (saveLotsOfFiles.isSelected()) {
					npairsFrame.saveLotsOfFiles = true;
					saveSplitFiles.setEnabled(true);
					rotatePCsToImgSpace.setEnabled(true);
				}
				else {
					npairsFrame.saveLotsOfFiles = false;
					saveSplitFiles.setSelected(false);
					saveSplitFiles.setEnabled(false);
					rotatePCsToImgSpace.setEnabled(false);
					rotatePCsToImgSpace.setSelected(false);
				}
			}
		});

		saveSplitFiles = new JCheckBoxMenuItem("Save split results");
		saveSplitFiles.setToolTipText("<html>If checked, results for each split will be saved.");
		saveSplitFiles.setEnabled(false); // only enabled if saveLotsOfFiles is checked
		saveSplitFiles.setSelected(false);

		saveSplitFiles.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (saveSplitFiles.isSelected()) {
					npairsFrame.saveSplitFiles = true;
				}
				else {
					npairsFrame.saveSplitFiles = false;
				}
			}
		});

		calcProcWithEigims = new JCheckBoxMenuItem("Use CV eigenimages in Procrustes calculation");
		calcProcWithEigims.setToolTipText("<html>If checked, Procrustes transformation for matching split " +
		"\nresults to reference results will be calculated on CV eigenimages instead of CV scores.");
		calcProcWithEigims.setEnabled(true);
		calcProcWithEigims.setSelected(false);

		calcProcWithEigims.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (calcProcWithEigims.isSelected()) {
					npairsFrame.calcProcWithEigims = true;
				}
				else {
					npairsFrame.calcProcWithEigims = false;
				}
			}
		});

		rotatePCsToImgSpace = new JCheckBoxMenuItem("Save PC eigenimages in image space");
		rotatePCsToImgSpace.setToolTipText("<html>If checked, PC eigenimages will be projected <br> "
				+ "back into original image space (time-consuming). <br> Default is to leave them in "
				+ "initial feature selection (EVD) space.");
		rotatePCsToImgSpace.setEnabled(false); // only enabled if saveLotsOfFiles is checked
		rotatePCsToImgSpace.setSelected(false);

		rotatePCsToImgSpace.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (rotatePCsToImgSpace.isSelected()) {
					npairsFrame.rotatePCsToImgSpace = true;
				}
				else {
					npairsFrame.rotatePCsToImgSpace = false;
				}
			}
		});

		saveDataPostPCA = new JCheckBoxMenuItem("Save denoised (post-PCA) input data");
		saveDataPostPCA.setToolTipText("<html>If checked, data from PCA passed into CVA <br> "
				+ "will be saved in original image space as 4D Nifti file.");
		saveDataPostPCA.setEnabled(true);
		saveDataPostPCA.setSelected(false);

		saveDataPostPCA.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (saveDataPostPCA.isSelected()) {
					npairsFrame.saveDataPostPCA = true;
				}
				else {
					npairsFrame.saveDataPostPCA = false;
				}
			}
		});


		useMatlabForInitEVD = new JCheckBoxMenuItem("Use Matlab for Initial EVD");
		useMatlabForInitEVD.setEnabled(true);

		if (npairsFrame.matlibType.toUpperCase().equals("MATLAB")) {
			useMatlabForInitEVD.setSelected(true);
			useMatlabForInitEVD.setEnabled(false);
		}

		useMatlabForInitEVD.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				if (useMatlabForInitEVD.isSelected()) {
					npairsFrame.matlibTypeForInitFeatSel = "MATLAB";
					//					if (debug) {
					//						System.out.println("Selecting Matlab for Init Feat Sel...");
					//					}
				}
				else {
					npairsFrame.matlibTypeForInitFeatSel = npairsFrame.matlibType;
					//					if (debug) {
					//						System.out.println("Selecting " + npairsFrame.matlibTypeForInitFeatSel
					//								+ " for Init Feat Sel...");
					//					}
				}
			}
		});

		///////Disabled for now///////
		useMatlabForInitEVD.setEnabled(false);
		//////////////////////////////

		optionMenu.add(saveLotsOfFiles);
		optionMenu.add(saveSplitFiles);
		optionMenu.add(rotatePCsToImgSpace);
		optionMenu.add(saveDataPostPCA);
		optionMenu.add(useMatlabForInitEVD);	
		optionMenu.add(calcProcWithEigims);
	}


	protected void updateDeselectMenuConds(NpairsAnalysisFrame parent,
			Vector<Integer> newClassSelection) {
		deselectCVAClasses.removeActionListener(deselectCVAClasses.getActionListeners()[0]);
		deselectCVAClasses.addActionListener(new DeselectClassListener(parent, newClassSelection,
				classNames));
	}


	protected void updateDeselectMenuConds(NpairsAnalysisFrame parent, 
			Vector<Integer> newClassSelection,
			String[] newClassNames) {
		deselectCVAClasses.removeActionListener(deselectCVAClasses.getActionListeners()[0]);
		deselectCVAClasses.addActionListener(new DeselectClassListener(parent, newClassSelection,
				newClassNames));
	}

	/**
	 * Update the selectable conditions by processing the passed in groups
	 * and determining the conditions held within.
	 * @param sessProfiles the groups of session files.
	 * @return true if successfull, false otherwise.
	 */
	protected boolean updateDeselectMenuConds(Vector <String[]> sessProfiles) {
		if (useCondsAsClasses == true) {  


			String[] classNames = NpairsAnalysisFrame.getConditionNamesFromGroups(sessProfiles);

			//an error occured while extracting condition information.
			if(classNames == null) return false; 

			Vector<Integer> selectAll = new Vector<Integer>(classNames.length);
			for (int i = 0; i < classNames.length; ++i) {
				selectAll.add(1);
			}

			updateDeselectMenuConds(npairsFrame, selectAll, classNames);
		}
		return true;
	}

	public void clearFields() {

		// Clear the session file info for each group.
		NpairsSessionProfilePanel sessionProfilesPanel = npairsFrame.npairsSessionProfilePanel;
		DefaultTableModel model = sessionProfilesPanel.model;

		int numRows = model.getRowCount();
		for (int i = numRows - 1; i != -1; i--) {
			model.removeRow(i);
		}
		sessionProfilesPanel.sessionProfiles.removeAllElements();
		sessionProfilesPanel.conditionSelection.removeAllElements();
		//sessionProfilesPanel.behaviorBlockConditionSelection.removeAllElements();

		// Reset the Npairs analysis type info.
		ModelOptionsPanel npairsOptionPanel = npairsFrame.npairsAnalysisTypeOptionsPanel;

		npairsOptionPanel.preprocOptions[0].setSelected(true);
		npairsOptionPanel.preprocOptions[0].setEnabled(true);
		npairsOptionPanel.modelOptions[0].setSelected(false);
		npairsOptionPanel.modelOptions[0].setEnabled(false);
		npairsOptionPanel.modelOptions[1].setSelected(true);
		npairsOptionPanel.modelOptions[2].setSelected(true);
		npairsOptionPanel.cvaClassFilenameField.setText(null);
		npairsOptionPanel.cvaClassFilenameField.setEnabled(false);
		// Also reset class info for Deselect menu
		classNames = null;
		updateDeselectMenuConds(npairsFrame, null);
		useCondsAsClasses = true;
		// Reset the resampling info.
		ResamplingPanel resamplingOptionsPanel = npairsFrame.resamplingOptionsPanel;

		resamplingOptionsPanel.resampChkBox.setSelected(true);
		resamplingOptionsPanel.splitHalfXValidButton.setSelected(true);
		resamplingOptionsPanel.bootstrapButton.setSelected(false);
		resamplingOptionsPanel.bootstrapButton.setEnabled(false);

		resamplingOptionsPanel.loadSplitsChkBox.setSelected(false);
		resamplingOptionsPanel.splitPartition[0].setEnabled(true);
		resamplingOptionsPanel.splitPartition[1].setEnabled(true);
		resamplingOptionsPanel.splitPartition[0].setText("");
		resamplingOptionsPanel.splitPartition[1].setText("");
		resamplingOptionsPanel.splitPartLabel.setEnabled(true);
		if (npairsFrame.isBlocked) {
			resamplingOptionsPanel.splitObjLabel.setEnabled(true);
			resamplingOptionsPanel.splitObjChoice.setEnabled(true);
		}
		else {
			resamplingOptionsPanel.splitObjLabel.setEnabled(false);
			resamplingOptionsPanel.splitObjChoice.setEnabled(false);
		}
		resamplingOptionsPanel.splitObjChoice.setSelectedIndex(0);
		resamplingOptionsPanel.numSplitsField.setEnabled(true);
		resamplingOptionsPanel.numSplitsField.setText("50");
		resamplingOptionsPanel.numSplitsLabel.setEnabled(true);
		resamplingOptionsPanel.splitsFileFieldLabel.setEnabled(false);
		resamplingOptionsPanel.splitsFileBrowseButt.setEnabled(false);
		resamplingOptionsPanel.splitsFilenameField.setEnabled(false);
		resamplingOptionsPanel.splitsFilenameField.setText("");

		// Reset the initial feature selection info.
		DataReductionOptionsPanel dataReductionOptionsPanel = npairsFrame.dataReducOptsPanel;

		//		dataReductionOptionsPanel.initFeatSelLabel.setSelected(false);
		dataReductionOptionsPanel.initEVDRegButton.setSelected(true);
		dataReductionOptionsPanel.loadEVDChkBox.setSelected(false);
		dataReductionOptionsPanel.evdFilePrefField.setText("");
		dataReductionOptionsPanel.evdFilePrefField.setEnabled(false);
		dataReductionOptionsPanel.evdBrowseButt.setEnabled(false);

		dataReductionOptionsPanel.reductionFactorField.setText("0.3");
		dataReductionOptionsPanel.pcsInSplitCVAField.setText("0");
		dataReductionOptionsPanel.pcsInFullDataCVAField.setText("0");
		dataReductionOptionsPanel.pcsInSplitCVAField.setEnabled(true);
		dataReductionOptionsPanel.pcsInFullDataCVAField.setEnabled(true);
		dataReductionOptionsPanel.rField.setText("");
		dataReductionOptionsPanel.sField.setText("");
		dataReductionOptionsPanel.rField.setEnabled(false);
		dataReductionOptionsPanel.sField.setEnabled(false);
		dataReductionOptionsPanel.rLab.setEnabled(false);
		dataReductionOptionsPanel.sLab.setEnabled(false);
		dataReductionOptionsPanel.setPCRange.setSelected(false);
		dataReductionOptionsPanel.setPCRange.setEnabled(true);
		dataReductionOptionsPanel.setAutoPCRange.setSelected(false);
		dataReductionOptionsPanel.setAutoPCRange.setEnabled(false);
		
		dataReductionOptionsPanel.fullDataMultFactField.setEnabled(false);
		dataReductionOptionsPanel.fullDataMultFactField.setText("1.0");
		dataReductionOptionsPanel.fullDataMultFactLab.setEnabled(false);

		// Reset the results file name.
		npairsFrame.resultsFilePrefField.setText(NpairsAnalysisFrame.DEFAULT_RESULT_FILE_PREF);

		//Reset persistence information for the loading and saving file choosers
		fileName = null;

		resetOptionsMenu();
	}

	private void resetOptionsMenu() {
		saveLotsOfFiles.setEnabled(true);
		saveLotsOfFiles.setSelected(false);
		npairsFrame.saveLotsOfFiles = false;
		saveSplitFiles.setEnabled(false);
		saveSplitFiles.setSelected(false);
		calcProcWithEigims.setSelected(false);
		npairsFrame.saveSplitFiles = false;
		rotatePCsToImgSpace.setSelected(false);
		rotatePCsToImgSpace.setEnabled(false);
		npairsFrame.rotatePCsToImgSpace = false;
		saveDataPostPCA.setSelected(false);
		//		saveDataPostPCA.setEnabled(true); // TODO decide: always enabled?
		npairsFrame.saveDataPostPCA = false;
		useMatlabForInitEVD.setSelected(false);
		npairsFrame.matlibTypeForInitFeatSel = npairsFrame.matlibType;

	}

	@Override
	public void save() {
		SaveNpairsSetupParamsActionListener nspActionListener = 
			new SaveNpairsSetupParamsActionListener(npairsFrame, fileName);	
		nspActionListener.save();
	}

	@Override
	/**
	 * saveAs will be called instead of save when save is hit if no 
	 * file has been loaded or saved yet. 
	 */
	public void saveAs(){
		//Initialize inheirited dialog  
		if(sChooser == null){
			sChooser = new JFileChooser();
			sChooser.addChoosableFileFilter(filter);
		}

		if(fileName == null){
			File path = new File(npairsFrame.resultsFilePrefField.getText());
			String defaultDir = path.getParent();
			String defaultName = path.getName();

			//Set the default name to use if the name prefix could not be found
			if(defaultName.equals("")){
				defaultName = "npairs_NPAIRSAnalysisSetup.mat";
			}else{
				defaultName += NpairsAnalysisSetupFileFilter.ext;
			}

			//set the directory the dialog should start in
			if(defaultDir == null){
				defaultDir = ".";
			}

			File dir = new File(defaultDir);
			if(!dir.exists()){
				sChooser.setCurrentDirectory(new File("."));
			}else{
				sChooser.setCurrentDirectory(dir);
			}

			sChooser.setSelectedFile(new File(defaultName));
			int opt = sChooser.showSaveDialog(npairsFrame);
			if(opt == JFileChooser.APPROVE_OPTION){
				fileName = sChooser.getSelectedFile().getAbsolutePath();
				String filterDesc = sChooser.getFileFilter().getDescription();
				if(filterDesc.equals("Npairs Analysis Setup Parameter Files") 
						&&!fileName.endsWith(NpairsAnalysisSetupFileFilter.ext)){
					fileName += NpairsAnalysisSetupFileFilter.ext;
				}
				save();

			}

		}else{
			File path = new File(fileName);
			String dir = path.getParent();
			String name = path.getName();

			if(dir != null){
				sChooser.setCurrentDirectory(new File(dir));
			}else{
				sChooser.setCurrentDirectory(new File("."));
			}
			sChooser.setSelectedFile(new File(name));

			int opt = sChooser.showSaveDialog(npairsFrame);
			if (opt == JFileChooser.APPROVE_OPTION){
				fileName = sChooser.getSelectedFile().getAbsolutePath();
				String filterDesc = sChooser.getFileFilter().getDescription(); 
				if(filterDesc.equals("Npairs Analysis Setup Parameter Files")
						&& !fileName.endsWith(NpairsAnalysisSetupFileFilter.ext)){
					fileName += NpairsAnalysisSetupFileFilter.ext;
				}
				save();
			}
		}

	}

	public void load(String loadThisFile){
		fileName = loadThisFile;
		load();
	}


	public void load() {
		// Get NPAIRS analysis setup parameters from file
		MLStructure npairsSetupInfo = null;
		try {
			npairsSetupInfo = (MLStructure)new NewMatFileReader(fileName).getContent().get("npairs_setup_info");
		} 
		catch (Exception ex) {
			JOptionPane.showMessageDialog(null, "NPAIRS analysis setup parameter file " + fileName 
					+ "\ncould not be loaded.", "Error", JOptionPane.ERROR_MESSAGE);
			clearFields();
			loadErr = true;
			return;
		}
	

		// Load NpairsSessionProfilePanel info
		MLStructure sessionFileInfo = (MLStructure)npairsSetupInfo.getField("session_file_info");
		int numGroups = sessionFileInfo.getN();
		Vector<String[]> sessionProfiles = new Vector<String[]>(numGroups);

		for (int g = 0; g < numGroups; ++g) {	
			String[] currSessFiles = MLFuncs.MLCell1dRow2StrArray((MLCell)(sessionFileInfo.
					getField("session_files", g)));
			sessionProfiles.add(g, currSessFiles);
		}
		npairsFrame.npairsSessionProfilePanel.setSessProfiles(sessionProfiles);
		
		
		// load NpairsResamplingPanel info
		boolean doSplitHalfXValid = (((MLDouble)npairsSetupInfo.getField("split_half_xvalid")).
				get(0,0).intValue() == 1);
		boolean doBootstrap = (((MLDouble)npairsSetupInfo.getField("bootstrap")).get(0,0).intValue()
				== 1);

		int numSplits = 0;
		String splitsInfoFilename = "";
		int[] splitPartition = null;
		String splitObjType = this.npairsFrame.getSplitObjTypes()[0]; // default value:
		// Session file

		if (doSplitHalfXValid || doBootstrap) { // do resampling
			numSplits = ((MLDouble)npairsSetupInfo.getField("num_splits")).
			get(0,0).intValue();
			try {
				splitsInfoFilename = ((MLChar)npairsSetupInfo.getField(
				"splits_info_filename")).getString(0);
			}
			catch (NullPointerException npe) {
				// no splits loaded from file
				try {
					splitPartition = new int[2];
					splitPartition[0] = ((MLDouble)npairsSetupInfo.getField(
					"split_partition")).get(0,0).intValue();
					splitPartition[1] = ((MLDouble)npairsSetupInfo.getField(
					"split_partition")).get(0,1).intValue();
				}
				catch (NullPointerException npe2) {
					// no split partition info saved in file
				}
			}

			try {
				splitObjType = ((MLChar)npairsSetupInfo.getField(
				"split_type")).getString(0);		
			}
			catch(NullPointerException e) {
				// split obj type not implemented yet; default is to use Session file
			}
		}

		if (!splitsInfoFilename.equals("") && !splitsInfoFilename.contains(".")) {
			splitsInfoFilename.concat(".vols");
		}
		npairsFrame.resamplingOptionsPanel.setResamplingOptions(doSplitHalfXValid, 
				doBootstrap, numSplits, splitsInfoFilename, splitPartition, splitObjType);

		// Load NpairsOptionsPanel info
		try {
			boolean doMSR = (((MLDouble)npairsSetupInfo.getField("do_msr")).get(0,0).intValue() == 1);
			npairsFrame.npairsAnalysisTypeOptionsPanel = npairsFrame.npairsAnalysisTypeOptionsPanel.setPreprocOptions(doMSR);
		}
		catch (NullPointerException npe) {
			// MSR wasn't implemented yet when the analysis setup file being read in was created
		}
		boolean doGLM = (((MLDouble)npairsSetupInfo.getField("do_glm")).get(0,0).intValue() == 1);
		boolean doPCA = (((MLDouble)npairsSetupInfo.getField("do_pca")).get(0,0).intValue() == 1);
		boolean doCVA = (((MLDouble)npairsSetupInfo.getField("do_cva")).get(0,0).intValue() == 1);

		npairsFrame.npairsAnalysisTypeOptionsPanel.setAnalysisOptions(doGLM, doPCA, doCVA, splitObjType);

		if (doCVA) {
			if (doPCA) {
				boolean normPCs = (((MLDouble)npairsSetupInfo.getField("norm_pcs")).get(0,0).intValue() == 1);
				npairsFrame.npairsAnalysisTypeOptionsPanel.normPCChkBox.setSelected(normPCs);
			}

			// Load class info
			MLDouble classSelectionInfo = (MLDouble)npairsSetupInfo.getField("class_selection");
			int numClasses = classSelectionInfo.getM();
			Vector<Integer> classSelection = new Vector<Integer>(numClasses);

			for (int i = 0; i < numClasses; ++i) {
				classSelection.add(classSelectionInfo.get(i).intValue());
			}
			npairsFrame.npairsSessionProfilePanel.setSelectedClasses(classSelection);

			String cvaClassFile = "";

			try {
				cvaClassFile = ((MLChar)npairsSetupInfo.getField("cva_class_file")).getString(0);
			}
			catch (NullPointerException npe) {
				// no class file flag in analysis file
			}
			npairsFrame.npairsAnalysisTypeOptionsPanel.cvaClassFilenameField.setText(cvaClassFile);

			if (cvaClassFile.length() > 0) {
				// Get class labels using classfile 
				useCondsAsClasses = false;
				try {
					classNames = getClassNames(cvaClassFile);
				}
				catch (IOException e) {
					JOptionPane.showMessageDialog(null, "Could not load NPAIRS class file [" + cvaClassFile + 
							"].", "Error", JOptionPane.ERROR_MESSAGE);
					//					clearFields();
					loadErr = true;
					//					return;
				}
			}

			else {
				// No classfile entered; set class labels using sessionfile conditions
				// (Assumption: all sessionfiles have same conditions)
				useCondsAsClasses = true;

				classNames = NpairsAnalysisFrame.getConditionNamesFromGroups(sessionProfiles);
				if (classNames == null){
					//					clearFields();
					loadErr = true;
					//					return; //error retrieving condition names.
				}

			}

			updateDeselectMenuConds(this.npairsFrame, classSelection);

		}



		// load DataReductionOptionsPanel info
		double dataRedFactor = ((MLDouble)npairsSetupInfo.getField("drf")).get(0,0);

		boolean doInitEVD = false;
		try {
			doInitEVD = (((MLDouble)npairsSetupInfo.getField("do_init_evd")).
					get(0,0).intValue() == 1);
		} 
		catch (NullPointerException npe) {
			// do_init_svd replaced with do_init_evd in v 1.1.6; bkwds 
			// compatibility kept
			doInitEVD = (((MLDouble)npairsSetupInfo.getField("do_init_svd")).
					get(0,0).intValue() == 1);
		}
		boolean doUnwghtedEVD = false;
		try {
			doUnwghtedEVD = (((MLDouble)npairsSetupInfo.getField("norm_init_evd")).
					get(0,0).intValue() == 1);
		}
		catch (NullPointerException npe) {
			// unwghted ("normed") evd wasn't implemented yet when current analysis file was created
		}

		boolean loadEVD = false;
		String evdFilePref = null;
		if (doInitEVD || doUnwghtedEVD) {
			try {
				loadEVD = (((MLDouble)npairsSetupInfo.getField("load_evd")).get(0,0).intValue() == 1);
				if (loadEVD) {
					evdFilePref = ((MLChar)npairsSetupInfo.getField("evd_file_prefix")).getString(0);
				}
			} 
			catch (NullPointerException npe) {
				// load_svd and svd_file_prefix replaced with load_evd and evd_file_prefix
				// in v 1.1.6; bkwds compatibility kept
				loadEVD = (((MLDouble)npairsSetupInfo.getField("load_svd")).get(0,0).intValue() == 1);
				if (loadEVD) {
					evdFilePref = ((MLChar)npairsSetupInfo.getField("svd_file_prefix")).getString(0);
				}			
			}

			String splitPCs = null;
			String fullDataPCs = null;
			String pcRange = null;
			int pcStep = 0;
			boolean setPCRange = false;
			boolean setAutoPCRange = false;
			double pcMultFact = 0.0;
			try {
				splitPCs = ((MLChar)npairsSetupInfo.getField("pcs_training")).getString(0);
				fullDataPCs = ((MLChar)npairsSetupInfo.getField("pcs_all_data")).getString(0);
			}
			catch(NullPointerException npe) {
				setPCRange = true;
				pcStep = ((MLDouble)npairsSetupInfo.getField("pc_step")).get(0,0).intValue();	
				try {
					setAutoPCRange = ((MLDouble)npairsSetupInfo.getField("set_auto_pc_range")).get(0,0).
						intValue() == 1;
				} catch (NullPointerException npe2) {
					// auto pc range wasn't option yet in this analysis setup file
				}
				if (!setAutoPCRange) {
					pcRange = ((MLChar)npairsSetupInfo.getField("pc_range")).getString(0);
					pcMultFact = ((MLDouble)npairsSetupInfo.getField("pc_mult_factor")).get(0,0).doubleValue();
				}
				else {
					// TODO: Set this in global variable somewhere instead
					pcMultFact = 1.0;
				}
				
			}

			//NOTE: !doUnwghtedEVD is used here instead of doRegularEVD to ensure that no files
			// with doRegularEVD set to false cause unweighted EVD to happen automatically, since
			// unweighted EVD was implemented after regular evd and it's possible (though unlikely) that
			// a user may have created a setup file with "do_init_svd" (i.e. regular EVD) set to false
			// in an attempt to perform no EVD at all. We don't want an unweighted EVD be performed
			// in that case. (This case could come up only because until now, doing any initial
			// feature selection was misleadingly offered as option (not requirement) in GUI, even
			// though doing no EVD has ever been thoroughly implemented or tested.) 
			// Of course, by using !doUnwghtedEVD, we get a regular EVD automatically in case
			// doUnWghtedEVD is set to false.  But this is the behaviour we actually want right now.  
			npairsFrame.dataReducOptsPanel.setDataReductionOptions(!doUnwghtedEVD, loadEVD, evdFilePref, 
					dataRedFactor, splitPCs, fullDataPCs, setPCRange, pcRange, pcStep, pcMultFact, setAutoPCRange);

			// Options menu 
			try {
				double d = ((MLDouble)npairsSetupInfo.getField("save_multi_files")).
				get(0,0).doubleValue();
				boolean saveMultiFiles = (d == 1);
				saveLotsOfFiles.setSelected(saveMultiFiles);

				double d2 = ((MLDouble)npairsSetupInfo.getField("save_split_results")).
				get(0,0).doubleValue();
				boolean saveSplits = (d2 == 1);
				saveSplitFiles.setSelected(saveSplits);

				double d3 = ((MLDouble)npairsSetupInfo.getField("pc_eigims_in_img_space")).
				get(0,0).doubleValue();
				boolean pcEigimIntoOrigSpc = (d3 == 1);
				rotatePCsToImgSpace.setSelected(pcEigimIntoOrigSpc);

				double d4 = ((MLDouble)npairsSetupInfo.getField("save_data_post_pca")).
				get(0,0).doubleValue();
				boolean saveDenoisedData = (d4 == 1);
				saveDataPostPCA.setSelected(saveDenoisedData);

				double d5 = ((MLDouble)npairsSetupInfo.getField("eigim_procrustes")).
				get(0,0).doubleValue();
				boolean useEigims = (d5 == 1);
				calcProcWithEigims.setSelected(useEigims);
			}

			catch (NullPointerException npe) { 
				// either 
				// (1) save_multi_files or save_split_results not included in current analysis 
				// setup file, but it's OK - will just not save them by default
				// or (2) pc_eigims_in_img_space, save_data_post_pca or eigim_procrustes not included in current 
				// analysis setup file, but it's OK, too - just don't project them into img space, save denoised 
				// (post-PCA) data or do Procrustes via eigims by default
			}

			String resultsFilename = ((MLChar)npairsSetupInfo.getField("results_filename")).getString(0);
			String resultsFilePref = resultsFilename;
			if (resultsFilename.endsWith(NpairsfMRIResultFileFilter.EXTENSION)) {
				resultsFilePref = resultsFilename.substring(0, resultsFilename.indexOf(
						NpairsfMRIResultFileFilter.EXTENSION));
			}
			npairsFrame.resultsFilePrefField.setText(resultsFilePref);		

			loadErr = false;
			return;
		}
	}

	
	public static String[] getConditionNames(String sessionFileName) throws FileNotFoundException, IOException {
		MLStructure sessProfStruct = (MLStructure)new NewMatFileReader(sessionFileName).
		getContent().get("session_info");
		String[] classNames = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
		return classNames;
	}


	public static String[] getClassNames(String cvaClassFile) throws IOException {
		int[] classLabels = NpairsjIO.readIntsFromFile(cvaClassFile);
		int[] uniqLabels = MLFuncs.sortAscending(MLFuncs.unique(classLabels));
		String[] classNames = new String[uniqLabels.length];
		for (int i = 0; i < uniqLabels.length; ++i) {
			classNames[i] = new Integer(uniqLabels[i]).toString();
		}
		return classNames;
	}

	/**
	 * 
	 * @return Error status upon loading an analysis setup file.
	 * True if there was an error, false otherwise.
	 */
	public boolean getLoadErr(){
		return loadErr;
	}

}


final class DeselectClassListener implements ActionListener {

	private NpairsAnalysisFrame parent;
	private Vector<Integer> classSelection;
	private String[] classNames;

	protected DeselectClassListener(NpairsAnalysisFrame parent, 
			Vector<Integer> classSelection, 
			String[] classNames) {
		this.parent = parent;
		this.classSelection = classSelection;	
		this.classNames = classNames;
	}

	public void actionPerformed(ActionEvent ae) {
		new DeselectClassFrame(parent, classSelection, classNames);	
		parent.setClassSelectUpdated(true);
		parent.npairsSessionProfilePanel.setSelectedClasses(classSelection);

	}
}


final class DeselectClassFrame extends JModalFrame {

	protected DeselectClassFrame(JFrame parent,
			Vector<Integer> classSelection,
			String[] classNames) {

		super("Deselect CVA Classes");
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(parent);

		if(classNames == null) { // no class info entered yet
			JOptionPane.showMessageDialog(null, "Class label information needs to be added\n" +
					"(via class file or, if no class file entered, using condition info in " +
					"session files)\nbefore you can deselect CVA classes.", 
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}

		setJMenuBar(new BaseMenuBar(this));
		add(new ConditionSelectionPanel(this, classNames, classSelection));

		//		 Display the window
		pack();
		setResizable(false);

		// Position the frame on the screen
		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		int x = (int)(screen.getWidth() - getWidth()) / 2;
		int y = (int)(screen.getHeight() - getHeight()) / 2;
		setLocation(x, y);

		setVisible(true);
	}
}


final class SaveNpairsSetupParamsActionListener implements ActionListener {

	private NpairsAnalysisFrame npairsAnalysisFrame;

	private NpairsAnalysisSetupFileFilter filter; 

	private String fileName = null;

	private String extension; 

	public SaveNpairsSetupParamsActionListener(NpairsAnalysisFrame npairsAnalysisFrame, String fileName) {
		this.npairsAnalysisFrame = npairsAnalysisFrame;
		this.fileName = fileName;
		if (npairsAnalysisFrame.isBlocked) {
			this.filter = new NpairsAnalysisSetupFileFilter();
			this.extension = "_NPAIRSAnalysisSetup.mat";
		}
		else {
			this.filter = new NpairsAnalysisSetupERFileFilter();
			this.extension = "_NPAIRSAnalysisSetupER.mat";
		}
	}

	public void actionPerformed(ActionEvent e) {
		String filePath = ".";
		File f = null;
		if (fileName != null) {
			f = new File(fileName);
			filePath = (f.getAbsolutePath());
		}

		JFileChooser chooser = new JFileChooser(filePath);
		chooser.setSelectedFile(f);

		if(filter != null) {
			chooser.setFileFilter(filter);
		}
		int option = chooser.showDialog(npairsAnalysisFrame, "Save As");
		if(option == JFileChooser.APPROVE_OPTION) {
			fileName = chooser.getSelectedFile().getAbsolutePath();
			if(extension != null && !chooser.getSelectedFile().getName().contains(".")) {
				fileName += extension;
			}
			save();
		}
	}

	/**
	 * Check all the fields in the npairs analysis frame to make sure they are
	 * correct. If correct save the analysis in a file,otherwise report the 
	 * relevant error. The filename is either the filename passed to the 
	 * constructor of this listener or it is the filename set through the 
	 * actionPerformed method.
	 */
	protected boolean save() {

		if(npairsAnalysisFrame.resultsFilePrefField.getText().trim().equals("")){
			JOptionPane.showMessageDialog(npairsAnalysisFrame,
					"The result file prefix must be filled in.",
					"Error",JOptionPane.ERROR_MESSAGE);
			return false;
		}
		// Has Deselect been updated since class info was last changed?
		if (!npairsAnalysisFrame.classSelectionUpdated()) {
			JOptionPane.showMessageDialog(npairsAnalysisFrame, 
					"Class selection has not been updated since class file info"
					+ " or session file info changed. \n" 
					+ "Please go to Deselect menu and choose which classes to include in analysis.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}

		int numGroups;
		int maxNumSessions = 0;
		Vector <String[]> sessionProfiles = npairsAnalysisFrame.npairsSessionProfilePanel.sessionProfiles;
		Vector <Integer> classSelection = npairsAnalysisFrame.npairsSessionProfilePanel.conditionSelection;
		numGroups = sessionProfiles.size();

		if (!(numGroups > 0)) {
			JOptionPane.showMessageDialog(npairsAnalysisFrame, 
					"You must include at least one sessionfile.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}

		if (!(classSelection.size() > 0)) {
			JOptionPane.showMessageDialog(npairsAnalysisFrame,
					"You must select which classes to include in analysis before " +
					"saving npairs setup file. \nPlease go to Deselect menu to choose classes.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}

		for (int i = 0; i < numGroups; ++i) {
			int currNumSessions = sessionProfiles.get(i).length;
			maxNumSessions = Math.max(maxNumSessions, currNumSessions);
		}

		MLStructure npairsSetupInfo = new MLStructure("npairs_setup_info", new int[] {1, 1});

		// If 'Event-related' analysis, datamats are loaded; if 'Blocked', data is read in
		// from image files
		int loadDatamats = 0;
		if (!npairsAnalysisFrame.isBlocked) {
			loadDatamats = 1;
		}
		npairsSetupInfo.setField("load_datamats", new MLDouble("loadDatamats", 
				new double[][]{{loadDatamats}}));

		// Save sessionfile info for each group
		MLStructure sessionFileInfo = new MLStructure("session_file_info", new int[] {1, numGroups});
		// TODO: refine test for valid split object partition info; should consider how splits
		// include proportion of data from each grp when analyzing multiple groups 
		// (Currently just check total number of input session files to determine valid split
		// partition entries.)
		//int nSessFiles = 0;
		for (int g = 0; g < numGroups; ++g) {
			int currNumSessions = sessionProfiles.get(g).length;
			//nSessFiles += currNumSessions;
			MLCell currSessionFiles = new MLCell("session_files" + g, new int[] {1, currNumSessions});
			for(int sf = 0; sf < currNumSessions; ++sf) {
				currSessionFiles.set(new MLChar("session_file" + sf, sessionProfiles.get(g)[sf]), 0, sf);
			}

			sessionFileInfo.setField("session_files", currSessionFiles, g);
			sessionFileInfo.setField("grp_size", new MLDouble("grp_size" + g, 
					new double[][]{{currNumSessions}}), g);
		}
		npairsSetupInfo.setField("session_file_info", sessionFileInfo);

		int numClasses = classSelection.size();
		MLDouble classSelect = new MLDouble("class_selection", new int[]{numClasses, 1});

		for (int i = 0; i < numClasses; ++i) {
			classSelect.set(new Double(classSelection.get(i)), i);
		}
		npairsSetupInfo.setField("class_selection", classSelect);

		// Save NPAIRS analysis type info
		int doMSR = 0;
		if (npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.preprocOptions[0].isEnabled()
				&&
				npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.preprocOptions[0].isSelected()) {
			doMSR = 1;
		}
		npairsSetupInfo.setField("do_msr", new MLDouble("do_msr", new double[][] {{doMSR}}));

		int doGLM = 0;
		if (npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.modelOptions[0].isSelected()) {
			doGLM = 1;
		}
		npairsSetupInfo.setField("do_glm", new MLDouble("do_glm", new double[][]{{doGLM}}));

		int doPCA = 0;
		if (npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.modelOptions[1].isSelected()) {
			doPCA = 1;
		}	
		npairsSetupInfo.setField("do_pca", new MLDouble("do_pca", new double[][]{{doPCA}}));

		int doCVA = 0;
		if (npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.modelOptions[2].isSelected()) {
			doCVA = 1;
			String cvaClassFile = npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.
			cvaClassFilenameField.getText();
			npairsSetupInfo.setField("cva_class_file", new MLChar("cva_class_file", cvaClassFile));
		}
		npairsSetupInfo.setField("do_cva", new MLDouble("do_cva", new double[][]{{doCVA}}));

		if (doCVA == 1 && doPCA == 1) {
			int normPCs = 0;
			if (npairsAnalysisFrame.npairsAnalysisTypeOptionsPanel.normPCChkBox.isSelected()) {
				normPCs = 1;
			}
			npairsSetupInfo.setField("norm_pcs", new MLDouble("norm_pcs", 
					new double[][]{{normPCs}}));
		}

		else {
			// for now, must do PCA + CVA
			JOptionPane.showMessageDialog(null, "NPAIRS Analysis Modelling Options: " +
					"\nSelect PCA + CVA.  This is the only NPAIRS analysis option " +
					"currently implemented.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}

		// Save resampling info
		if (npairsAnalysisFrame.resamplingOptionsPanel.resampChkBox.isSelected() 
				&&
				npairsAnalysisFrame.resamplingOptionsPanel.splitHalfXValidButton.isSelected()) {
			npairsSetupInfo.setField("split_half_xvalid", new MLDouble(
					"split_half_xvalid", new double[][]{{1}}));
		}
		else {
			npairsSetupInfo.setField("split_half_xvalid", new MLDouble("split_half_xvalid",
					new double[][]{{0}}));
		}
		if (npairsAnalysisFrame.resamplingOptionsPanel.resampChkBox.isSelected()
				&&
				npairsAnalysisFrame.resamplingOptionsPanel.bootstrapButton.isSelected()) {
			npairsSetupInfo.setField("bootstrap", new MLDouble("bootstrap",
					new double[][]{{1}}));
		}
		else {
			npairsSetupInfo.setField("bootstrap", new MLDouble("bootstrap",
					new double[][]{{0}}));
		}

		if (npairsAnalysisFrame.resamplingOptionsPanel.resampChkBox.isSelected()) {
			int numSplits = Integer.parseInt(npairsAnalysisFrame.resamplingOptionsPanel.
					numSplitsField.getText());
			npairsSetupInfo.setField("num_splits", new MLDouble("num_splits", 
					new double[][]{{numSplits}}));

			if (npairsAnalysisFrame.resamplingOptionsPanel.loadSplitsChkBox.isSelected()) {
				String splitsInfoFilename = npairsAnalysisFrame.resamplingOptionsPanel.
				splitsFilenameField.getText();
				npairsSetupInfo.setField("splits_info_filename", new MLChar("splits_info_filename",
						splitsInfoFilename));
			}
			else {
				setSplitPartition(npairsSetupInfo);
			}

			String splitObjType = (String)npairsAnalysisFrame.resamplingOptionsPanel.
			splitObjChoice.getSelectedItem();
			npairsSetupInfo.setField("split_type", new MLChar("split_type", splitObjType));
		}

		// Save initial feature selection info
		if (npairsAnalysisFrame.dataReducOptsPanel.initEVDRegButton.isSelected()) {
			npairsSetupInfo.setField("do_init_evd", new MLDouble("do_init_evd", 
					new double[][]{{1}})); 
			npairsSetupInfo.setField("norm_init_evd", new MLDouble("norm_init_evd", 
					new double[][]{{0}}));
		}
		else if (npairsAnalysisFrame.dataReducOptsPanel.initEVDNormedButton.isSelected()) {
			npairsSetupInfo.setField("do_init_evd", new MLDouble("do_init_evd", 
					new double[][]{{1}})); 
			npairsSetupInfo.setField("norm_init_evd", new MLDouble("norm_init_evd", 
					new double[][]{{1}})); 
		}		
		else {
			npairsSetupInfo.setField("do_init_evd", new MLDouble("do_init_evd", 
					new double[][]{{0}})); 
			npairsSetupInfo.setField("norm_init_evd", new MLDouble("norm_init_evd", 
					new double[][]{{0}})); 
		}

		if (npairsAnalysisFrame.dataReducOptsPanel.loadEVDChkBox.isSelected()) {
			npairsSetupInfo.setField("load_evd", new MLDouble("load_evd", 
					new double[][]{{1}}));
			String svdFilePrefix = npairsAnalysisFrame.dataReducOptsPanel.
			evdFilePrefField.getText();
			if (svdFilePrefix.length() == 0) {
				JOptionPane.showMessageDialog(npairsAnalysisFrame,
						"Please enter EVD " +
						"file prefix to load EVD information.", 
						"Warning", JOptionPane.ERROR_MESSAGE);
				return false;
			}
			npairsSetupInfo.setField("evd_file_prefix", new MLChar("evd_file_prefix", 
					svdFilePrefix));
		}
		else {
			npairsSetupInfo.setField("load_evd", new MLDouble("load_evd", 
					new double[][]{{0}}));
		}

		double dataReductFactor = Double.parseDouble(npairsAnalysisFrame.dataReducOptsPanel.
				reductionFactorField.getText());
		npairsSetupInfo.setField("drf", new MLDouble("drf", new double[][]{{dataReductFactor}}));

		boolean setPCRange = npairsAnalysisFrame.dataReducOptsPanel.
			setPCRange.isSelected();
		if (doPCA == 1 && doCVA == 1) {
			// save pc info for passing into cva
			//			boolean setPCRange = npairsAnalysisFrame.dataReducOptsPanel.
			//					setPCRange.isSelected();
			if (!setPCRange) {	
				String pcsForSplit = npairsAnalysisFrame.dataReducOptsPanel.
				pcsInSplitCVAField.getText();
				String pcsForFullData = npairsAnalysisFrame.dataReducOptsPanel.
				pcsInFullDataCVAField.getText();	
				npairsSetupInfo.setField("pcs_training", new MLChar("pcs_training",
						pcsForSplit));
				npairsSetupInfo.setField("pcs_all_data", new MLChar("pcs_all_data",
						pcsForFullData));
			}
			else {
				int setAutoPCRange = 0;
				if (npairsAnalysisFrame.dataReducOptsPanel.setAutoPCRange.
						isSelected()) setAutoPCRange = 1;
				npairsSetupInfo.setField("set_auto_pc_range", new MLDouble(
					"set_auto_pc_range",new double[][]{{setAutoPCRange}}));	
				try {
					int pcStep = Integer.parseInt(npairsAnalysisFrame.dataReducOptsPanel.
							sField.getText());
					System.out.println("PC step as int: " + pcStep);
					npairsSetupInfo.setField("pc_step", new MLDouble("pc_step", 
							new double[][]{{pcStep}}));
				} catch (NumberFormatException nfe) {
					JOptionPane.showMessageDialog(null, "Must include PC step size (as integer).",
							"Error", JOptionPane.ERROR_MESSAGE);
					return false;
				}
				
				if (setAutoPCRange == 0) {
					
					String pcRange = npairsAnalysisFrame.dataReducOptsPanel.rField.getText();
					try {
					double pcMultFact = Double.parseDouble(npairsAnalysisFrame.dataReducOptsPanel.
							fullDataMultFactField.getText());
					npairsSetupInfo.setField("pc_mult_factor", new MLDouble("pc_mult_factor", 
							new double[][]{{pcMultFact}}));
					if (pcMultFact <= 0) {
						JOptionPane.showMessageDialog(null, "Proportion of PC range to use in full-data" +
								" CVA must be greater than zero.",
								"Error", JOptionPane.ERROR_MESSAGE);
						return false;
					}
					} catch (NumberFormatException nfe) {
						JOptionPane.showMessageDialog(null, "Must include proportion of PCs to use in full data.",
								"Error", JOptionPane.ERROR_MESSAGE);
					}
					npairsSetupInfo.setField("pc_range", new MLChar("pc_range", pcRange));
					
				}
			}
		}

		// save lots of results files?
		double saveMulti = 1;
		if (!npairsAnalysisFrame.saveLotsOfFiles) {
			saveMulti = 0;
		}
		npairsSetupInfo.setField("save_multi_files", new MLDouble("save_multi_files",
				new double[][] {{saveMulti}}));
		double saveSplits = 0;
		if (npairsAnalysisFrame.saveSplitFiles) {
			saveSplits = 1;
		}
		npairsSetupInfo.setField("save_split_results", new MLDouble("save_split_results",
				new double[][] {{saveSplits}}));

		double useEigims = 0;
		if(npairsAnalysisFrame.calcProcWithEigims) {
			useEigims = 1;
		}
		npairsSetupInfo.setField("eigim_procrustes", new MLDouble("eigim_procrustes",
				new double[][] {{useEigims}}));

		double pcEigimIntoOrigSpace = 0;
		if (npairsAnalysisFrame.rotatePCsToImgSpace) {
			pcEigimIntoOrigSpace = 1;
		}
		npairsSetupInfo.setField("pc_eigims_in_img_space", new MLDouble("pc_eigims_in_img_space",
				new double[][] {{pcEigimIntoOrigSpace}}));

		double saveDataPostPCA = 0;
		if (npairsAnalysisFrame.saveDataPostPCA) {
			saveDataPostPCA = 1;
			if (setPCRange) {
				int OK = JOptionPane.showConfirmDialog(null, "Denoised data will be saved for each PC setting. "
						+ "\nYou will need a lot of disk space!  \nContinue saving?", "Warning", 
						JOptionPane.OK_CANCEL_OPTION);
				if (OK != JOptionPane.OK_OPTION) {
					return false;
				}
			}
		}
		npairsSetupInfo.setField("save_data_post_pca", new MLDouble("save_data_post_pca",
				new double[][] {{saveDataPostPCA}}));

		npairsSetupInfo.setField("results_filename", new MLChar("results_filename", 
				npairsAnalysisFrame.resultsFilePrefField.getText() + 
				NpairsfMRIResultFileFilter.EXTENSION));

		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(npairsSetupInfo);
		try {
			new MatFileWriter(fileName, list);
		}
		catch(Exception ex) {
			JOptionPane.showMessageDialog(npairsAnalysisFrame,
					"Could not save to npairs setup file " + fileName + ".", 
					"Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}
		return true;
		//		if (setFilename) {
		//			npairsAnalysisFrame.setNpairsSetupParamsFileName(fileName);
		//		}
	}

	private void setSplitPartition(MLStructure npairsSetupInfo) {

		String splitPartStr[] = new String[2];
		splitPartStr[0] = npairsAnalysisFrame.resamplingOptionsPanel.
		splitPartition[0].getText();
		splitPartStr[1] = npairsAnalysisFrame.resamplingOptionsPanel.
		splitPartition[1].getText();
		String splitPartErrorMessage = "Invalid split partition values.";

		try {
			int[] splitPartition = new int[2];
			splitPartition[0] = Integer.parseInt(splitPartStr[0]);
			splitPartition[1] = Integer.parseInt(splitPartStr[1]);
			if (splitPartition[0] <= 0 || splitPartition[1] <= 0) {
				JOptionPane.showMessageDialog(null, splitPartErrorMessage,
						"Error", JOptionPane.ERROR_MESSAGE);
			}

			String splitObjType = (String)npairsAnalysisFrame.resamplingOptionsPanel.
			splitObjChoice.getSelectedItem();
			int nSplitObj = npairsAnalysisFrame.resamplingOptionsPanel.
			getNumSplitObj(splitObjType);
			if (splitPartition[0] + splitPartition[1] > nSplitObj) {
				JOptionPane.showMessageDialog(null, splitPartErrorMessage,
						"Error", JOptionPane.ERROR_MESSAGE);
			}
			npairsSetupInfo.setField("split_partition", new MLDouble("split_partition",
					new double[][]{{splitPartition[0], splitPartition[1]}}));
		}
		catch (NumberFormatException nfe) {
			JOptionPane.showMessageDialog(null, splitPartErrorMessage,
					"Error", JOptionPane.ERROR_MESSAGE);
		}
		catch (IOException ioe) {
			JOptionPane.showMessageDialog(null, ioe.getMessage(),
					"Error", JOptionPane.ERROR_MESSAGE);
		}
	}
}