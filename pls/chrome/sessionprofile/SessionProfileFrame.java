package pls.chrome.sessionprofile;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.IOException;
import java.io.PipedOutputStream;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.MissingResourceException;
import java.util.Vector;
import java.util.Enumeration;

import javax.swing.*;
import javax.swing.border.Border;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

import pls.sessionprofile.RunGenerateDatamat;
import pls.sessionprofile.RunInformation;

import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.FilePathCheck;

import pls.shared.MLFuncs;
import pls.shared.AnalyzeNiftiFileFilter;
import pls.shared.NpairsBlockSessionFileFilter;
import pls.shared.StreamedProgressDialog;
import pls.shared.StreamedProgressHelper;
import pls.shared.fMRISessionFileFilter;
import pls.shared.BfMRISessionFileFilter;


import npairs.io.NpairsjIO;
import java.util.NoSuchElementException;


@SuppressWarnings("serial")
public class SessionProfileFrame extends JFrame {
    
    private SessionProfileFrame frame = null;
    
    private SessionProfileMenuBar menubar = null;
	
	private boolean initialFile = true; //see loadImage()
	
    private JLabel numConditionsLabel = new JLabel("Number of Conditions: 0");
    
    private JLabel numRunsLabel = new JLabel("Number of Runs: 0");

    private BrainViewerPanel brainPanel = new BrainViewerPanel();

	protected boolean isNPAIRSSessionProfile = false;

	protected JTextField descriptionField = new JTextField();
	
	protected JTextField datamatPrefixField = new JTextField();
    
	protected ButtonGroup mergeDataGroup = new ButtonGroup();
    
	// {Condition Name, Reference Scan Onset, Number of Reference Scans}
	protected Vector<String[]> conditionInfo = new Vector<String[]>();
    
	protected Vector<RunInformation> runInfo = new Vector<RunInformation>();
	
    protected JTextField brainRegionFileField = new JTextField();
	
    protected JTextField brainRegionThresholdField = new JTextField("0.15");
    
 //   protected JRadioButton[] brainRegionRadioButtons = new JRadioButton[2];
    
    protected JRadioButton[] brainRegionRadioButtons;
    
    protected JTextField scansField = new JTextField("0");
	
    protected JTextField runsField = new JTextField();
	
    protected JTextField slicesField = new JTextField();
	
    protected JTextField windowField = new JTextField("8");
	
    protected JCheckBox normalizeVolumeCheckBox = new JCheckBox("Normalize data with volume mean");
	
    protected JCheckBox normalizeScansCheckBox = new JCheckBox("Normalize data with reference scans");
	
//    protected JCheckBox meanScanRemovalCheckBox = new JCheckBox("Remove mean session scan from each volume");
    
    protected JCheckBox singleSubjectCheckBox = new JCheckBox("Single Subject Analysis");
    
    protected JButton createStDatamatButton;
    
    protected boolean isBlockedFmri = false;

	private String lastPath = null;
	
	private Thread datamatGenerator;
	
	public SessionProfileFrame(final boolean isBlockedFmri, final boolean isNPAIRSAnalysis,
			String mainMenuBarTitle) {
		
		super(mainMenuBarTitle);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		
	    frame = this;
	    this.isBlockedFmri = isBlockedFmri;
	    this.isNPAIRSSessionProfile = isNPAIRSAnalysis;   

	    brainRegionRadioButtons = new JRadioButton[3];

	    // Session File Panel
	    Border border;
		JPanel sessionFilePanel = buildSessionFilePanel(isBlockedFmri);
		
		// Brain region panel
		JPanel brainRegionPanel = buildBrainRegionPanel();

		// Datamat panel
		JPanel datamatPanel = buildDatamatPanel();
	    
		// Buttons
		JPanel buttonPane = createButtons();
        
        //Add panels
        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("Session File", sessionFilePanel);
        tabs.addTab("Brain Region", brainRegionPanel);
        if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
        	tabs.addTab("Spatial-Temporal Datamat", datamatPanel);
        }
        else {
        	tabs.addTab("Datamat Setup Info", datamatPanel);
        }
        tabs.addTab("Sample Image Orientation", brainPanel);
		
		JPanel mainPanel = new JPanel(new BorderLayout(5, 5));
		mainPanel.add(tabs, BorderLayout.NORTH);
		mainPanel.add(buttonPane, BorderLayout.SOUTH);
	    border = BorderFactory.createEtchedBorder();
	    border = BorderFactory.createEmptyBorder(5, 5, 5, 5);
	    mainPanel.setBorder(border);
		
        add(mainPanel);
	    
        // Display the window
        pack();
        
        // Position the frame on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = (int)(screen.getWidth() - getWidth()) / 2;
	    int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setResizable(false);
        
        setVisible(true);
	}
	
	protected void clearFields() {
		descriptionField.setText(null);
		datamatPrefixField.setText(null);
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			mergeDataGroup.getElements().nextElement().setSelected(true);
		}
		conditionInfo.removeAllElements();
		runInfo.removeAllElements();
		numConditionsLabel.setText("Number of Conditions: 0");
		numRunsLabel.setText("Number of Runs: 0");
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			brainRegionRadioButtons[0].setSelected(true); // use thresh
		}      
		else {
			brainRegionRadioButtons[1].setSelected(true); // use mask
		}
		brainRegionFileField.setText("");
		brainRegionThresholdField.setText("0.15");
		scansField.setText("0");
		runsField.setText("");
		slicesField.setText("");
		windowField.setText("8");
		normalizeVolumeCheckBox.setSelected(false);
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			normalizeScansCheckBox.setSelected(true);
		}
		singleSubjectCheckBox.setSelected(false);
		brainPanel.removeImages();
	}

	/**
	 * Create the buttons for the 'session file' tab.
	 * @return A panel containing the buttons.
	 */
	private JPanel createButtons() {
		JButton clearSessionButton = new JButton("Clear Session", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Refresh16.gif")));
        clearSessionButton.setIconTextGap(15);
        clearSessionButton.addActionListener(new ActionListener(){
        	public void actionPerformed(ActionEvent e) {
        		clearFields();
        	}});
		
        createStDatamatButton = new JButton("Create Spatial-Temporal Datamat", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
        JButton saveSessionInfoButton = new JButton("Save session setup info", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
        
        createStDatamatButton.setIconTextGap(15);
        
        menubar = new SessionProfileMenuBar(this);
	    setJMenuBar(menubar);
	    
        if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
	        createStDatamatButton.addActionListener(new RunGenerateDatamatActionListener(this));
    	}    
        else {
        	saveSessionInfoButton.addActionListener(new ActionListener() {
        		public void actionPerformed(ActionEvent e) {
        			//System.out.println("You pressed the 'save session info' button!");
        			
        			// check required info filled out in GUI
 /*       			boolean guiInfoOK = true;
        			// Session File panel
        			if (conditionInfo.size() < 2) {
        				JOptionPane.showMessageDialog(null, "Must enter at least 2 conditions", 
        						"Error", JOptionPane.ERROR_MESSAGE);
        				guiInfoOK = false;
        			}
        			if (runInfo.size() == 0) {
        				JOptionPane.showMessageDialog(null, "Must enter at least 1 run", 
        						"Error", JOptionPane.ERROR_MESSAGE);
        				guiInfoOK = false;
        			}
        			//Brain region panel
        			if (brainRegionFileField.getText().equals("")) {
        				JOptionPane.showMessageDialog(null, "Must enter mask filename", 
        						"Error", JOptionPane.ERROR_MESSAGE);
        				guiInfoOK = false;
        			}
        			if (guiInfoOK) {*/
        			try{
        					menubar.saveAs();
        			}catch(MissingResourceException e2){}
        			//}
        			
        		}
        	});
        	//TODO: add saveSessionInfoButton ActionListener
        }
		JPanel buttonPane = new JPanel(new GridLayout(1, 0));
        buttonPane.add(clearSessionButton);
        if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
        	buttonPane.add(createStDatamatButton);
        }
        else {
        	buttonPane.add(saveSessionInfoButton);
        }
		return buttonPane;
	}

	/**
	 * Create the session file panel.
	 * @param isBlockedFmri
	 * @return the session file panel.
	 */
	private JPanel buildSessionFilePanel(final boolean isBlockedFmri) {
		JLabel descriptionLabel = new JLabel("Session Description");
	    descriptionField.setColumns(12);

	    JLabel prefixLabel;
	    if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
	    	prefixLabel = new JLabel("<html><span style=\"color: red;\">*</span> Datamat Prefix</html>");
	    }
	    else {
	    	prefixLabel = new JLabel("");
	    }
	    
	    datamatPrefixField.setColumns(12);
	  
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 5, 5));
        labelPane.add(descriptionLabel);
        labelPane.add(prefixLabel);
        
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 5, 5));
        fieldPane.add(descriptionField);
        if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
        	fieldPane.add(datamatPrefixField);
        }
        else {
        	fieldPane.add(new JLabel(""));
        }
	    JPanel topPane = new JPanel(new GridLayout(0, 2));
        topPane.add(labelPane);
        topPane.add(fieldPane);
        
        JRadioButton[] mergeDataRadioButtons = new JRadioButton[2];
        
        if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
        	mergeDataRadioButtons[0] = new JRadioButton("Across All Runs");
        	mergeDataRadioButtons[0].setSelected(true);
        	mergeDataRadioButtons[1] = new JRadioButton("Within Each Run");
        
        	mergeDataGroup.add(mergeDataRadioButtons[0]);
        	mergeDataGroup.add(mergeDataRadioButtons[1]);
		}
		
		JPanel radioBox = new JPanel(new GridLayout(1, 0, 100, 0));
        
	    Border border = BorderFactory.createEtchedBorder();
	    radioBox.setBorder(border);
		
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			radioBox.add(mergeDataRadioButtons[0]);
			radioBox.add(mergeDataRadioButtons[1]);
		}
		
        JButton editConditionsButton = new JButton("Edit Conditions", new ImageIcon(this.getClass().
        		getResource("/toolbarButtonGraphics/general/Edit16.gif")));
        editConditionsButton.setIconTextGap(15);
        editConditionsButton.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) { 
        	new EditConditionsFrame(frame, (isBlockedFmri && isNPAIRSSessionProfile)); }});
		
        JButton editRunsButton = new JButton("Edit Runs", new ImageIcon(this.getClass().getResource(
        		"/toolbarButtonGraphics/general/Edit16.gif")));
        editRunsButton.setIconTextGap(15);
        editRunsButton.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) {
        	new EditRunsFrame(frame); }});

	    labelPane = new JPanel(new GridLayout(0, 1, 100, 0));
        labelPane.add(numConditionsLabel);
        labelPane.add(numRunsLabel);
        
        fieldPane = new JPanel(new GridLayout(0, 1, 100, 0));
        fieldPane.add(editConditionsButton);
        fieldPane.add(editRunsButton);
        
	    JPanel runConditionPane = new JPanel(new GridLayout(0, 2));
	    runConditionPane.add(labelPane, BorderLayout.CENTER);
	    runConditionPane.add(fieldPane, BorderLayout.LINE_END);
        
	    JPanel sfPanel = new JPanel(new GridLayout(0, 1, 20, 20));
	    sfPanel.add(topPane);
	    sfPanel.add(radioBox);
	    sfPanel.add(runConditionPane);
	    border = BorderFactory.createEmptyBorder(50, 0, 0, 0);
	    sfPanel.setBorder(border);
	    JPanel sessionFilePanel = new JPanel();
		sessionFilePanel.add(sfPanel);
		return sessionFilePanel;
	}
	
	private JPanel buildBrainRegionPanel(){
		ButtonGroup group;
		JPanel predefinedPanel;
		JPanel definedPanel;
		JPanel brainRegionPanel;
		JPanel brPanel;
		Border border;
		
		final JButton browseButton;
		final JLabel thresholdLabel;
		
		brainRegionRadioButtons[0] = new JRadioButton(
				"Define brain region automatically:");
		brainRegionRadioButtons[1] = new JRadioButton(
				"Use predefined brain region:");
		
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			brainRegionRadioButtons[2] = new JRadioButton(
					"Include all voxels >= 0 in brain mask");
		}

		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			brainRegionRadioButtons[0].setSelected(true);
			brainRegionFileField.setEnabled(false);
		}
		else {
			//TODO: reenable when thresholding is NPAIRS block option
			brainRegionRadioButtons[0].setEnabled(false);
			brainRegionThresholdField.setEnabled(false);
			brainRegionRadioButtons[1].setSelected(true);
		}
		
		group = new ButtonGroup();
		group.add(brainRegionRadioButtons[0]);
		group.add(brainRegionRadioButtons[1]);
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			//TODO: enable for block npairs when implemented
			group.add(brainRegionRadioButtons[2]);
		}

		brainRegionFileField.setColumns(25);

		browseButton = new JButton("Select mask file", 
				new ImageIcon(this.getClass().
				getResource("/toolbarButtonGraphics/general/Open16.gif")));
		browseButton.setIconTextGap(15);
		
		browseButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new AnalyzeNiftiFileFilter());
				int option = chooser.showOpenDialog(SessionProfileFrame.this);
				if (option == JFileChooser.APPROVE_OPTION) {
					brainRegionFileField.setText(
							chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
		
		predefinedPanel = new JPanel();
		predefinedPanel.add(brainRegionFileField);
		predefinedPanel.add(browseButton);

		thresholdLabel = new JLabel("Threshold");
		brainRegionThresholdField.setColumns(5);
		
		definedPanel = new JPanel();
		definedPanel.add(thresholdLabel);
		definedPanel.add(brainRegionThresholdField);
		
		ItemListener activateBehaviour = new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				if(brainRegionRadioButtons[1].isSelected()) { // load mask
					brainRegionFileField.setEnabled(true);
					browseButton.setEnabled(true);
					thresholdLabel.setEnabled(false);
					brainRegionThresholdField.setEnabled(false);
				} 
				else {
					brainRegionFileField.setEnabled(false);
					browseButton.setEnabled(false);		
					if (brainRegionRadioButtons[0].isSelected()) { // use thresh
						thresholdLabel.setEnabled(true);
						brainRegionThresholdField.setEnabled(true);
					}
					else { // incl all vox >= 0 
						thresholdLabel.setEnabled(false);
						brainRegionThresholdField.setEnabled(false);
						brainRegionThresholdField.setText("0.00");
					}
				}
			}
		};

		// TODO: fix listener; right now switching from one button to another means
		// the listener is called more than once
		brainRegionRadioButtons[0].addItemListener(activateBehaviour);
		brainRegionRadioButtons[1].addItemListener(activateBehaviour);
		
		brPanel = new JPanel(new GridLayout(0,2,5,10));
		brPanel.add(brainRegionRadioButtons[0]);
		brPanel.add(definedPanel);
		brPanel.add(brainRegionRadioButtons[1]);
		border = BorderFactory.createEmptyBorder(50, 0, 0, 0);
	    brPanel.setBorder(border);
        
	    Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = (int)(screen.getWidth() - getWidth()) / 2;
	    int y = (int)(screen.getHeight() - getHeight()) / 2;
	    
	    JPanel con = new JPanel(new BorderLayout());
	    con.add(brPanel, BorderLayout.NORTH);
	    con.add(predefinedPanel, BorderLayout.CENTER);
	    if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
	    	con.add(brainRegionRadioButtons[2], BorderLayout.SOUTH);
	    }
	    con.setSize(new Dimension(x/2,y/2));
	    
	    brainRegionPanel = new JPanel();
	    brainRegionPanel.add(con);

	    return brainRegionPanel;
		
	}
	
	/**
	 * Create the datamat panel (tab) for the session viewer.
	 * @return the datamat panel.
	 */
	private JPanel buildDatamatPanel(){
		JPanel widgets = new JPanel(new GridLayout(0,2,5,10));
	    
		scansField.setColumns(10);
		runsField.setColumns(10);
		slicesField.setColumns(10);
		slicesField.setEnabled(false); // re-enable when it works
		windowField.setColumns(10);
		
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			normalizeScansCheckBox.setSelected(true);
		}
		
		widgets.add(new JLabel("Number of scans to be skipped"));
		widgets.add(scansField);
		widgets.add(new JLabel("Runs to be ignored"));
		widgets.add(runsField);
		
	    if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
	    	JLabel slicesLabel = new JLabel("Slices to be ignored");
	    	slicesLabel.setEnabled(false); // TODO: re-enable when it works
		    widgets.add(slicesLabel);
		    widgets.add(slicesField);
	    }
	    
	    if(!isBlockedFmri) {
	    	widgets.add(new JLabel("Temporal window size (in scans)"));
	    	widgets.add(windowField);
	    }
	    
	    if (isBlockedFmri && isNPAIRSSessionProfile) {
	    	normalizeVolumeCheckBox.setEnabled(false);
	    	normalizeScansCheckBox.setEnabled(false);
	    	singleSubjectCheckBox.setEnabled(false);
	    }
	    widgets.add(normalizeVolumeCheckBox);
	    widgets.add(new JLabel(""));
	    
	    widgets.add(normalizeScansCheckBox);
	    widgets.add(new JLabel(""));
	    
	    singleSubjectCheckBox.setEnabled(false); // TODO: re-enable when sing. subj. datamats have been tested
	    widgets.add(singleSubjectCheckBox);
	    widgets.add(new JLabel(""));
	    
	    JPanel datamatPanel = new JPanel();
	    datamatPanel.add(widgets);
	    Border border = BorderFactory.createEmptyBorder(50,0,0,0);
	    datamatPanel.setBorder(border);
	    
	    return datamatPanel;
	    
	}
	
	public String getSessionFilename() {
		return menubar.fileName;
	}
	
	public String getDatamatPrefix(){
		return datamatPrefixField.getText();
	}
	
	public void updateConditions(Vector<String[]> newConditionInfo) {
		conditionInfo = newConditionInfo;
		numConditionsLabel.setText("Number of Conditions: " + conditionInfo.size());
	}
	
	/**
	 * Called when a file is loaded or Store and Continue is hit in
	 * the Edit runs frame. Loads the sample brain images and sets
	 * the databrowser's field if it has never been set before.
	 * @param newRunInfo
	 */
	public void updateRuns(Vector<RunInformation> newRunInfo) {
		String fileName;
		int numRuns = newRunInfo.size();

		runInfo = newRunInfo;
		numRunsLabel.setText("Number of Runs: " + numRuns);

		if(numRuns > 0) {
			String dir;
			File directory = new File(runInfo.get(0).dataDirectory);
			//String imageDirectory = directory.getName();

			//first time a file is loaded we set the databrowser's field.
			if(getLastPath() == null){
				setLastPath(directory.getParent());
			}
			
			if(!checkDataPaths(false)) return;
			
			/*append the image directory to the path chosen by the user
			 in the data path browser (disabled)*/
			//dir = getLastPath() + File.separator + imageDirectory;
			dir = directory.getAbsolutePath();
			
			//first run's images are always the ones that get displayed?
			fileName = runInfo.get(0).dataFiles.split(" ")[0];
			
			//updateDatapaths(newRunInfo,dir);
			loadImage(dir, fileName);
		}
	}
	
	public boolean loadImage(String directory, String fileName) {
		return brainPanel.loadImage(directory, fileName);
	}

	/**
	 * Get the value of lastPath. The 'last path' is the last location selected
	 * by the user in the data browser. By default this value is the cwd if the
	 * first loaded file's data path is incorrect or it is its data path if it
	 * is correct.
	 *
	 * @return the value of lastPath
	 */
	public String getLastPath() {
		return lastPath;
	}

	/**
	 * Set the value of lastPath
	 *
	 * @param lastPath new value of lastPath
	 */
	public void setLastPath(String lastPath) {
		this.lastPath = lastPath;
	}

	public boolean mergeAcrossRuns() {
		// mergeRuns is set to true by default when creating NPAIRS session profile
		boolean mergeRuns = true;
		if (!(isBlockedFmri && isNPAIRSSessionProfile)) {
			mergeRuns = mergeDataGroup.getElements().nextElement().isSelected();
		}
		return mergeRuns;
	}
	
	/**
	 * Get the run information from a matlab structure into a Vector of 
	 * our RunInformation class
	 * @param runs The run information in a matlab structure
	 * @param isBlockedFmri If it is blocked fmri or not
	 * @return The run information as a Vector of our RunInformation class
	 * @throws IllegalArgumentException if event-related Session file is loaded when it should be Block
	 *  or vice versa
	 *   
	 */
	public static Vector<RunInformation> getRunInformation(MLStructure runs, boolean isBlockedFmri) {

		Vector<RunInformation> runInfo = new Vector<RunInformation>();
		String errorMessage = "Cannot load Block session file into Event-related session file window.";
		if (isBlockedFmri) {
			errorMessage = "Cannot load Event-related session file into Block session file window.";
		}		
		
		for (int j = 0; j < runs.getSize(); ++j) {
			String dataPath = ((MLChar)runs.getField("data_path", j)).getString(0);
			
			MLCell dataFilesML = (MLCell)runs.getField("data_files", j);
			String dataFiles = "";
			for (MLArray m : dataFilesML.cells()) {
				dataFiles += ((MLChar)m).getString(0) + " ";
			}
			dataFiles = dataFiles.trim();

			String onsetFieldName = "evt_onsets";
			if (isBlockedFmri) {
				onsetFieldName = "blk_onsets";
			}			

			try {
				MLCell cOnsets = (MLCell)runs.getField(onsetFieldName, j);
				ArrayList<String> eventOnsets = new ArrayList<String>();
				int nConds = cOnsets.getN();
				for (int k = 0; k < nConds; ++k) {
					MLDouble dCurrOnsets = (MLDouble)cOnsets.get(k);
					String currOnsets = "";
					int nOnsets = dCurrOnsets.getM();
					int nCols = dCurrOnsets.getN();
					if (nCols > 1) { 	  // onset vector orientation has been
						nOnsets = nCols;  // flipped in session file	
					}
					for(int i = 0; i < nOnsets; i++) {
						currOnsets += dCurrOnsets.get(i).intValue() + " ";
					}
					eventOnsets.add(currOnsets.trim());
				}

				RunInformation currRunInfo = new RunInformation(dataPath, dataFiles, eventOnsets);		

				if (isBlockedFmri) {
					MLCell cLengths = (MLCell)runs.getField("blk_length", j);
					ArrayList<String> eventLengths = new ArrayList<String>();
					for(int i = 0; i < cLengths.getSize(); i++) {
						String length = "";
						MLDouble dLengths = (MLDouble)cLengths.get(i);
						int nLengths = dLengths.getM();
						int nCols = dLengths.getN();
						if (nCols > 1) { 	    // length vector orientation has been
							nLengths = nCols;	// flipped in session file
						}
						for(int k = 0; k < nLengths; k++) {
							length += dLengths.get(k).intValue() + " ";
						}
						eventLengths.add(length.trim());
					}
					currRunInfo.lengths = eventLengths;
				}
				
				runInfo.add(currRunInfo);
			}
			catch (NullPointerException npe) {
				throw new IllegalArgumentException(errorMessage);
			}
			catch (ClassCastException cce) {
				throw new IllegalArgumentException(errorMessage);
			}
			catch (NoSuchElementException nsee) {		
				throw new IllegalArgumentException(errorMessage);
			}
		}
			
		return runInfo;
	}
	
	/**
	 * The user must enter in runs if they want to save the session file.
	 * @return true if there are runs to save, false otherwise.
	 */
	boolean checkRunsEmpty(){
		if(runInfo.isEmpty()){
			JOptionPane.showMessageDialog(this,"Please provide run information before " +
					"saving the file.","Error", JOptionPane.ERROR_MESSAGE);
			return true;
		}
		return false;
	}
	/**
	 * Check that the data paths for the runs are correct.
	 * @param rInfo run information.
	 * @param suppress suppress the warning message if true
	 * @return false if any of the paths are incorrect, true otherwise.
	 */
	boolean checkDataPaths(boolean suppress){
		String imgDir;
		File examineFile;
		
		for(RunInformation run : runInfo){
			imgDir = run.dataDirectory;
			for(String file : run.dataFiles.split(" ")){
				examineFile = new File(imgDir + File.separator + file);
				if(!examineFile.isFile()){
					if(!suppress){
						JOptionPane.showMessageDialog(this,
								"Image data path seems to be incorrect.",
								"Error", JOptionPane.ERROR_MESSAGE);
					}
					return false;
				}
			}
		}
		return true;
	}
	
	/**
	 * Checks if the path to the mask file is correct.
	 * @param path path to the mask file.
	 * @return true if the path exists, false otherwise.
	 */
	boolean checkMaskPath(){
		String path = brainRegionFileField.getText();
		File filePath = new File(path);
		
		if(!filePath.isFile()){
			JOptionPane.showMessageDialog(this, "Warning: " +
					"Mask data path seems to be incorrect.", "Warning", 
					JOptionPane.WARNING_MESSAGE);
			return false;
		}
		return true;
	}
	
	/**
	 * Update all runs to all use the same parent directory. 
	 * (Not currently used).
	 * @param newRunInfo
	 * @param dir
	 */
	private void updateDatapaths(Vector<RunInformation> newRunInfo, String dir){
		String imageDir;
		String parentDir = (new File(dir)).getParent();
		setLastPath(parentDir);
		
		for(RunInformation run : newRunInfo){
			imageDir = new File(run.dataDirectory).getName();
			run.dataDirectory = parentDir + File.separator + imageDir;
		}
	}
	
	boolean usingDatamats(){
		return (isNPAIRSSessionProfile 
				&& !isBlockedFmri) ||
				(isNPAIRSSessionProfile == false);
	}
	
	/**
	 * @return Return the session file name without its prefix.
	 */
	String getPrefixWithoutSuffix(){
		return menubar.prefixWithoutSuffix();
	}
	
	/**
	 * Call the menubar saveAs() function 
	 * @return true if the user did not cancel the save as dialog and 
	 * the session file path's prefix now matches the datamat prefix.
	 * 
	 */
	boolean saveAs(){
		try{
			menubar.saveAs();
			return true; 
		}catch (MissingResourceException e){
			return false;
		}
		
	}

	public void setDatamatGenerator(Thread datamatGenerator) {
		this.datamatGenerator = datamatGenerator;
	}

	public Thread getDatamatGenerator() {
		return datamatGenerator;
	}
}

@SuppressWarnings("serial")
final class SessionProfileMenuBar extends BaseSaveMenuBar {
	
	private SessionProfileFrame sessionFrame = null;
		
	public SessionProfileMenuBar(SessionProfileFrame sessionFrame) {
		super(sessionFrame);
		
		this.sessionFrame = sessionFrame;
		
		if (sessionFrame.isNPAIRSSessionProfile) {
			if (sessionFrame.isBlockedFmri) {
				setFileFilter(new NpairsBlockSessionFileFilter(),
						NpairsBlockSessionFileFilter.EXTENSION);
			} else {
				setFileFilter(new fMRISessionFileFilter(),
						fMRISessionFileFilter.EXTENSION);
			}
		} else {
			if (sessionFrame.isBlockedFmri) {
				setFileFilter(new BfMRISessionFileFilter(),
						BfMRISessionFileFilter.EXTENSION);
			} else {
				setFileFilter(new fMRISessionFileFilter(),
						fMRISessionFileFilter.EXTENSION);
			}
		}
		
        // Build the edit menu
        JMenu editMenu = new JMenu("Edit");
        editMenu.setMnemonic('E');
        editMenu.getAccessibleContext().setAccessibleDescription("Edit session");
        add(editMenu, 1);
        
        JMenuItem changeDataPaths = new JMenuItem("Change Data Paths");
        changeDataPaths.setMnemonic('C');
        changeDataPaths.getAccessibleContext().setAccessibleDescription("Change the paths within the session files");
        changeDataPaths.addActionListener(new ChangeDataPathsListener(sessionFrame));
        editMenu.add(changeDataPaths);
        
        //Remove the save as option
        //getMenu(0).remove(2);
        
//        JMenuItem mergeConditions = new JMenuItem("Merge Conditions");
//        mergeConditions.setMnemonic('M');
//        mergeConditions.getAccessibleContext().setAccessibleDescription("Merge the Conditions");
//        mergeConditions.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) { JOptionPane.showMessageDialog(null, "TODO: You're out of luck", "Merge Conditions", JOptionPane.PLAIN_MESSAGE); }});
//        editMenu.add(mergeConditions);
		
	}

	@Override
	public void load() {
		sessionFrame.clearFields();
		// Get needed variables from file
		MLStructure sessionInfo = null;
		
		try {
			sessionInfo = (MLStructure) new NewMatFileReader(fileName).getContent().get("session_info");
		} catch (Exception ex) {
			JOptionPane.showMessageDialog(null, "Session file " + fileName + " could not be loaded.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		sessionFrame.descriptionField.setText(((MLChar)sessionInfo.getField("description")).getString(0));
		
		// datamat not used in npairs block but included by default for compatibility with PLS
		sessionFrame.datamatPrefixField.setText(((MLChar)sessionInfo.getField("datamat_prefix")).getString(0));
		
		
		int acrossRun = 0;
		
		if (((MLDouble)sessionInfo.getField("across_run")) != null) {
			acrossRun = ((MLDouble)sessionInfo.getField("across_run")).get(0, 0).intValue();
		}
		
		if (!(sessionFrame.isBlockedFmri && sessionFrame.isNPAIRSSessionProfile)) {
			// set merge runs info in GUI
			Enumeration<AbstractButton> buttons = sessionFrame.mergeDataGroup.getElements();
			if(acrossRun == 1) {
				buttons.nextElement().setSelected(true);
				buttons.nextElement().setSelected(false);
			} 
			else {
				buttons.nextElement().setSelected(false);
				buttons.nextElement().setSelected(true);
			}

		}
		
		String[] conditions;
		MLCell conditionBaseline;
		
		// If there is no condition0, try to get the information from condition
		// Note that this check is only necessary for some hand-made session profiles.
		if ((MLCell)sessionInfo.getField("condition0") != null) {
			conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition0"));
		} else {
			conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
		}
		
		// If there is no condition_baseline0, try to get the information from condition_baseline
		// Note that this check is only necessary for some hand-made session profiles.
		if ((MLCell)sessionInfo.getField("condition_baseline0") != null) {
			conditionBaseline = (MLCell)sessionInfo.getField("condition_baseline0");
		} else {
			conditionBaseline = (MLCell)sessionInfo.getField("condition_baseline");
		}
		
	    Vector<String[]> conditionInfo = new Vector<String[]>();
	    boolean defaultRefInfo = true;
	    for(int i = 0; i < conditions.length; i++) {
	    	String conditionName = conditions[i];
	    	MLDouble baseline = (MLDouble)conditionBaseline.get(0, i);
	    	String refScanOnset = new Integer(baseline.get(0, 0).intValue()).toString();
	    	String numRefScans = new Integer(baseline.get(0, 1).intValue()).toString();
	    	
	    	if (!refScanOnset.equals("0") || !numRefScans.equals("1")) {
	    		defaultRefInfo = false;  
	    		// TODO: allow custom ref info
	    		refScanOnset = "0";
	    		numRefScans = "1";   		
	    	}

	    	conditionInfo.add(new String[]{conditionName, refScanOnset, numRefScans});
	    }
	    if (!defaultRefInfo) {
	    	// only default ref scan info (0 1) has been tested so complain if anything 
    		// else is loaded in
    		// TODO: re-enable custom ref info when it works	    
	    String refSettingWarnMess = "Condition Info: Custom reference scan settings cannot " +
	        "currently be used.\nSetting reference scan onset and block size to default " +
	        "values 0 and 1 instead.";
	    JOptionPane.showMessageDialog(null, refSettingWarnMess, "Warning", 
	    		JOptionPane.WARNING_MESSAGE);
	    }
		sessionFrame.updateConditions(conditionInfo);
		
		try {
			Vector<RunInformation> runInfo = SessionProfileFrame.getRunInformation(
				(MLStructure)sessionInfo.getField("run"), sessionFrame.isBlockedFmri);
			sessionFrame.updateRuns(runInfo);
		}
		catch (IllegalArgumentException iae) {
			JOptionPane.showMessageDialog(null, iae.getMessage(), "Error", 
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		
		boolean missingBasicDmatInfo = false;
		boolean missingNormWindInfo = false;
		// mask info
		boolean inclAllVox = false;
		if (((MLDouble)sessionInfo.getField("use_all_voxels")) != null) {
			inclAllVox = (((MLDouble)sessionInfo.getField("use_all_voxels")).get(
					0, 0).intValue() == 1);	
			sessionFrame.brainRegionThresholdField.setText("0.00");			
		}
		if (!inclAllVox) {
			try {
				String maskFile = ((MLChar)sessionInfo.getField("mask")).getString(0);
				if (maskFile.length() > 0) {
					sessionFrame.brainRegionFileField.setText(maskFile);
					sessionFrame.brainRegionRadioButtons[1].setSelected(true);
				}
			}
			catch (NullPointerException e) {
				// sessionFrame.brainRegionFileField.setText("");
				try {
					Double thresh = ((MLDouble)sessionInfo.getField("brain_coord_thresh")).
						get(0);
					String pattern = "0.00";	
					DecimalFormat formatter = (DecimalFormat)DecimalFormat.getInstance();
					formatter.applyPattern(pattern);
					String formattedThr = formatter.format(thresh);
					//TODO: allow this for NPAIRS block, too
					if (!(sessionFrame.isBlockedFmri && sessionFrame.isNPAIRSSessionProfile)) {
						sessionFrame.brainRegionThresholdField.setText(formattedThr);
						sessionFrame.brainRegionRadioButtons[0].setSelected(true);
					}
					else { throw e; } // just until NPAIRS block allows thresh, too
				}
				
				catch (NullPointerException e2) {
					// "brain_coord_thresh" wasn't being saved yet when this session file
					// was created
					missingBasicDmatInfo = true;
//					if (sessionFrame.isNPAIRSSessionProfile && sessionFrame.isBlockedFmri) {
						JOptionPane.showMessageDialog(null,"Warning: Missing brain region info.",
								"Warning", JOptionPane.WARNING_MESSAGE);
//					}
				}
			}
		}
		else if (sessionFrame.isBlockedFmri && sessionFrame.isNPAIRSSessionProfile) {
			JOptionPane.showMessageDialog(null,"Warning: Missing Brain Region info.",
					"Warning", JOptionPane.WARNING_MESSAGE);
		}
		else { // use all voxels >= 0
			// TODO: enable this for NPAIRS block
			sessionFrame.brainRegionRadioButtons[2].setSelected(true);
			
		}
			
//		if (sessionFrame.isNPAIRSSessionProfile && sessionFrame.isBlockedFmri) {
			// datamat setup info
			try {
				int scansSkipped = ((MLDouble)sessionInfo.getField("scans_skipped")).get(0,0).intValue();
				sessionFrame.scansField.setText(Integer.toString(scansSkipped));
				String runsSkipped = ((MLChar)sessionInfo.getField("runs_skipped")).getString(0);
				sessionFrame.runsField.setText(runsSkipped);
			}
			catch (NullPointerException e) {
				missingBasicDmatInfo = true;
				JOptionPane.showMessageDialog(null, "Warning: Missing datamat setup info."						
						,"Warning",
						JOptionPane.WARNING_MESSAGE);
			}
			
			// try loading the rest of the datamat setup info (not used in NPAIRS block but
			// saved anyway)
			try {
				boolean volMeanNorm = ((MLDouble)sessionInfo.getField("normalize_volume_mean")).
						get(0,0).intValue() == 1;
				boolean refScanNorm = ((MLDouble)sessionInfo.getField("normalize_with_baseline")).
						get(0,0).intValue() == 1;
				boolean singSubjAnalysis = ((MLDouble)sessionInfo.getField("single_subject_analysis")).
					get(0,0).intValue() == 1;
				sessionFrame.normalizeVolumeCheckBox.setSelected(volMeanNorm);
				sessionFrame.normalizeScansCheckBox.setSelected(refScanNorm);
				sessionFrame.singleSubjectCheckBox.setSelected(singSubjAnalysis);
				
				if (!sessionFrame.isBlockedFmri) {
					int windowSz = ((MLDouble)sessionInfo.getField("temporal_window_size")).
						get(0,0).intValue();
					sessionFrame.windowField.setText(Integer.toString(windowSz));
				}
				
			}
			catch (NullPointerException npe) {
				// normalization and window info not saved in session file yet when this one was created
				// (or it's from Matlab)
				if (!(sessionFrame.isBlockedFmri && sessionFrame.isNPAIRSSessionProfile)) {
					missingNormWindInfo = true;
					//sessionFrame.normalizeScansCheckBox.setSelected(false);
					sessionFrame.windowField.setText("");
					JOptionPane.showMessageDialog(null, "Warning: Missing datamat normalization and/or " +
							"window size info."						
							,"Warning", JOptionPane.WARNING_MESSAGE);
				}
				
			}
			
			if (missingNormWindInfo) {
				String npairsBlockMess = "";
				if (sessionFrame.isBlockedFmri && missingBasicDmatInfo) {
					// missing stuff needed for NPAIRS block, too
					npairsBlockMess = "or running a Block NPAIRS analysis ";
				}
				JOptionPane.showMessageDialog(null, "Please fill in missing information " +
						"and save the session file before creating a datamat " +
						npairsBlockMess + "using PLSNPAIRS." +
						"\n(Files will remain compatible with Matlab PLS.)",				
						"Warning", JOptionPane.WARNING_MESSAGE);
			}
			else {
				// will never have norm info saved but not other ("basic") datamat info
			}

			
//		}
		
		/**
		 * If this is an event related npairs session file or a pls file then
		 * set the current prefix to be the newly loaded file's path with the
		 * suffix stripped.
		 */
//		if(sessionFrame.usingDatamats()){
//			fileName = prefixWithoutSuffix();
//		}
	}
		
	@Override
	public void saveAs(){
		if(sChooser == null){
			sChooser = new JFileChooser(filePath);
			sChooser.setPreferredSize(new Dimension(640,480));
		}

		if(filter != null) {
			sChooser.setFileFilter(filter);
		}
		
		if(sessionFrame.usingDatamats()){
			String datamatPrefix = sessionFrame.datamatPrefixField.getText();
						
			if(datamatPrefix == null || datamatPrefix.equals("")){
				JOptionPane.showMessageDialog(sessionFrame,
						"Datamat session prefix must be set before saving.");
				
				throw new MissingResourceException("Missing datamat prefix.",
													null,
													null);
			}
						
			disableText(sChooser,datamatPrefix);
		}
		
		int option = sChooser.showDialog(sessionFrame, "Save As");
		if(option == JFileChooser.APPROVE_OPTION) {
			fileName = sChooser.getSelectedFile().getAbsolutePath();
		
			if(extension != null && !fileName.endsWith(extension)) {
				fileName += extension;
			}
			save();
		}else{
			if(sessionFrame.usingDatamats()){
				throw new MissingResourceException("Saving cancelled.",
													null,
													null);
			}
		}
	}
		
	@Override
	public void save() {
		//////////Begin error checking////////////
		//Save the session file if the mask path or the data paths
		//are incorrect but stop datamat generation from taking place if either
		//of these errors occur.
		boolean stopDatamatGeneration = false;
		if(sessionFrame.usingDatamats()){
			checkFilePath(); //throws MRE exception.
		}
		
		//Ignore this error when saving and not generating a datamat.
		if(sessionFrame.brainRegionRadioButtons[1].isSelected()){
			if(!sessionFrame.checkMaskPath()){
				stopDatamatGeneration = true;
			}
		}
		
		//Ignore this error when saving and not generating a datamat.
		if(!sessionFrame.checkDataPaths(true)){
			JOptionPane.showMessageDialog(sessionFrame,
					"The Image datapath is incorrect.",
					"Warning",JOptionPane.WARNING_MESSAGE);
			stopDatamatGeneration = true;
		}
		
		if(sessionFrame.checkRunsEmpty()){
			throw new MissingResourceException("Runs empty",null,null);
		}
		
		if(sessionFrame.conditionInfo.size() < 2){
			JOptionPane.showMessageDialog(sessionFrame,
					"Must enter at least 2 conditions","Error",
					JOptionPane.ERROR_MESSAGE);
			throw new MissingResourceException("Need at least two conditions",
					null,null);
		}
		/////////////////////////////////////////
		
		
		MLStructure sessionInfo = new MLStructure("session_info", new int[] {1, 1});
		
		sessionInfo.setField("description", new MLChar("description", sessionFrame.descriptionField.getText()));
		
		// Save pls_data_path and other pls-specific variables even in npairs session file
		// so that session files are compatible.
		// TODO should also save npairs-specific variables (i.e., skipped scans and runs info)
		// in java-generated pls session files; that way, they will be totally compatible with
		// npairs, but should still be backwards compatible with plsgui in matlab, since extra
		// npairs variables should just be ignored when read in by plsgui
		
		String dataPath = ChangeDataPathsFrame.getCommonPath(sessionFrame.runInfo);
		sessionInfo.setField("pls_data_path", new MLChar("pls_data_path", dataPath));
		String dmPrefix = null;
		if (!sessionFrame.isNPAIRSSessionProfile) {
			dmPrefix = sessionFrame.datamatPrefixField.getText(); 
			if (dmPrefix.length() == 0) {
				JOptionPane.showMessageDialog(null, "Error - must enter datamat prefix " +
						"before saving session file", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
		else {
			// it's an NPAIRS analysis; create default datamat prefix using session file save prefix
			int extInd = this.fileName.indexOf(this.extension);
			int parentEnd = this.fileName.lastIndexOf(System.getProperty("file.separator"));
			dmPrefix = this.fileName.substring(parentEnd + 1, extInd);

//			System.out.println("Session filename: " + this.fileName);
//			System.out.println("dmPrefix: " + dmPrefix);	
		}
	
		sessionInfo.setField("datamat_prefix", new MLChar("datamat_prefix", dmPrefix));

		// set condition info for single run ('num_conditions0', 'conditions0', 'condition_baseline0')
        int numConds = sessionFrame.conditionInfo.size();
        int numRuns = sessionFrame.runInfo.size();

        MLCell conditions0 = new MLCell("conditions0", new int[] {1, numConds});
        MLCell conditionBaseline0 = new MLCell("condition_baseline0", new int[] {1, numConds});
        MLDouble numConditions0 = new MLDouble("num_conditions0", new double[][] {{numConds}});
        
        boolean defaultRefInfo = true;
        for(int i = 0; i < numConds; i++) {
        	String conditionName = sessionFrame.conditionInfo.get(i)[0];
        	double refScanOnset = new Double(sessionFrame.conditionInfo.get(i)[1]).doubleValue();
        	double numRefScans = new Double(sessionFrame.conditionInfo.get(i)[2]).doubleValue();
//        	//TODO: allow custom ref scan onset/block size info when they work
//        	if (refScanOnset != 0 || numRefScans != 1) {
//        		defaultRefInfo = false;
//        		refScanOnset = 0;
//        		numRefScans = 1;
//        	}
        	conditions0.set(new MLChar("condition0" + i, conditionName), 0, i);
        	conditionBaseline0.set(new MLDouble("condition_baseline0" + i, 
        			new double[][]{{refScanOnset, numRefScans}}), 0, i);
        }
//        //TODO: allow custom ref scan onset/block size info when they work
//        if (!defaultRefInfo) {
//        	String refSettingSaveMess = "Condition Info: Custom reference scan settings cannot " +
//        	 "currently be used.\nSaving session file with default reference scan onset and block size " +
//        	 "values 0 and 1 instead.";
//        	JOptionPane.showMessageDialog(null, refSettingSaveMess, "Warning", 
//        			JOptionPane.WARNING_MESSAGE);
//        }
        sessionInfo.setField("num_conditions0", numConditions0);
        sessionInfo.setField("condition0", conditions0);
        sessionInfo.setField("condition_baseline0", conditionBaseline0);
        
        
        // set condition info for all conditions/runs ('num_conditions', 'conditions', 'condition_baseline')
        MLCell conditions = conditions0;
        MLCell conditionBaseline = conditionBaseline0;
        MLDouble numConditions = numConditions0;
        
        if (!sessionFrame.mergeAcrossRuns()) {
        	// 'num_conditions', 'conditions' and 'condition_baseline' treat same condition
        	// in different runs as separate conditions
        	numConditions = new MLDouble("num_conditions", new double[][] {{numConds * numRuns}});
        	conditions = new MLCell("conditions", new int[] {1, numRuns * numConds});
        	conditionBaseline = new MLCell("condition_baseline", new int[] {1, numRuns * numConds});
        	int count = 0;
        	for(int i = 0; i < numRuns; i++) {
        		for(int j = 0; j < numConds; j++, count++) {
        			String conditionName = "Run" + (i + 1) + sessionFrame.conditionInfo.
        			get(j)[0];
        			double refScanOnset = new Double(sessionFrame.conditionInfo.get(j)[1]).
        			doubleValue();
        			double numRefScans = new Double(sessionFrame.conditionInfo.get(j)[2]).
        			doubleValue();
        			conditions.set(new MLChar("condition" + count, conditionName), 0, count);
        			conditionBaseline.set(new MLDouble("conditions_baseline" + count, 
        					new double[][]{{refScanOnset, numRefScans}}), 0, count);
        		}
        	}
        }
        
        sessionInfo.setField("num_conditions", numConditions);
        sessionInfo.setField("condition", conditions);
        sessionInfo.setField("condition_baseline", conditionBaseline);
        
	    // set run info
        sessionInfo.setField("num_runs", new MLDouble("num_runs", new double[][]{{numRuns}}));
		
		MLStructure run = new MLStructure("run", new int[]{1, numRuns});
		for(int i = 0; i < numRuns; i++) {
			String[] files = sessionFrame.runInfo.get(i).dataFiles.split(" ");
			int ns = files.length;
			run.setField("num_scans", new MLDouble("num_scans", new double[][]{{ns}}), i);
			run.setField("data_path", new MLChar("data_path", sessionFrame.runInfo.get(i).
					dataDirectory), i);
			MLCell cFiles = new MLCell("data_files" + i, new int[]{files.length, 1});
			for(int j = 0; j < files.length; j++) {
				cFiles.set(new MLChar("data_file" + j, files[j]), j, 0);
			}
			run.setField("data_files", cFiles, i);
			
			int extensionPos = files[0].lastIndexOf(".");
			String extension = "";
			if (extensionPos > -1) { // extension exists 
				extension = files[0].substring(extensionPos);
			}
			run.setField("file_pattern", new MLChar("file_pattern", "*" + extension));
			
			String onsetVariableName = "evt_onsets";
			if(sessionFrame.isBlockedFmri) {
				onsetVariableName = "blk_onsets";
			}
			
			MLCell cEventOnsets = new MLCell(onsetVariableName + i, new int[]{1, numConds});
			for(int j = 0; j < numConds; j++) {
				double[][] dEventOnsets = null;
				ArrayList<String> onsets = sessionFrame.runInfo.get(i).onsets;
				if (onsets.size() != numConds) {
					JOptionPane.showMessageDialog(null, "Must enter onset information " +
							"for every condition.", "Error", JOptionPane.ERROR_MESSAGE);
					throw new MissingResourceException("Missing onset info", 
							null, null);
					
				}
				String[] sEventOnsets = onsets.get(j).split(" ");
				for(String s : sEventOnsets) {
					dEventOnsets = MLFuncs.append(dEventOnsets, 
							new double[][]{{new Double(s).doubleValue()}});
				}
				cEventOnsets.set(new MLDouble(onsetVariableName + j, dEventOnsets), 0, j);
			}
			run.setField(onsetVariableName, cEventOnsets, i);
			
			if(sessionFrame.isBlockedFmri) {
				MLCell cEventLengths = new MLCell("blk_length" + i, new int[]{1, numConds});
				for(int j = 0; j < numConds; j++) {
					double[][] dEventLengths = null;
					ArrayList<String> lengths = sessionFrame.runInfo.get(i).lengths;
					if (lengths.size() != numConds) {
						JOptionPane.showMessageDialog(null, "Must enter block length information " +
								"for every condition.", "Error", JOptionPane.ERROR_MESSAGE);
						throw new MissingResourceException("Missing length info",
								null, null);
					}
					String[] sEventLengths = sessionFrame.runInfo.get(i).lengths.get(j).split(" ");
					for(String s : sEventLengths) {
						dEventLengths = MLFuncs.append(dEventLengths, 
								new double[][]{{new Double(s).doubleValue()}});
					}
					cEventLengths.set(new MLDouble("blk_length" + j, dEventLengths), 0, j);
				}
				run.setField("blk_length", cEventLengths, i);
			}
		}
		sessionInfo.setField("run", run);
		

		if (sessionFrame.mergeAcrossRuns()) {
			sessionInfo.setField("across_run", new MLDouble("across_run", new double[][]{{1}}));
		} 
		else { 
				sessionInfo.setField("across_run", new MLDouble("across_run", 
						new double[][]{{0}}));
			}
		//}
		
		if	(!sessionFrame.brainRegionFileField.getText().trim().equals("")
				&& sessionFrame.brainRegionRadioButtons[1].isSelected()) {
			sessionInfo.setField("mask", new MLChar("mask", sessionFrame.brainRegionFileField.
					getText()));
			sessionInfo.setField("use_all_voxels", new MLDouble("use_all_voxels", new double[][] {{0}}));
		}
		else if (sessionFrame.brainRegionRadioButtons[0].isSelected()) {
			// use threshold
			double coordThresh = new Double(sessionFrame.brainRegionThresholdField.getText()).doubleValue();
			sessionInfo.setField("brain_coord_thresh", new MLDouble("brain_coord_thresh", 
					new double[][]{{coordThresh}}));
			sessionInfo.setField("use_all_voxels", new MLDouble("use_all_voxels", new double[][] {{0}}));
		}
		else {
			// include all voxels >= 0
			sessionInfo.setField("use_all_voxels", new MLDouble("use_all_voxels", new double[][] {{1}})); 
			sessionInfo.setField("brain_coord_thresh", new MLDouble("brain_coord_thresh", 
					new double[][]{{0}}));
		}
		
		// add datamat setup info to session file (required for npairs)
		int skippedScans = Integer.parseInt(sessionFrame.scansField.getText());
		sessionInfo.setField("scans_skipped", new MLDouble("scans_skipped",
				new double[][] {{skippedScans}}));
		String skippedRunsText = sessionFrame.runsField.getText();
		if (skippedRunsText.equals("")) {
			skippedRunsText = "0";
		}
		sessionInfo.setField("runs_skipped", new MLChar("runs_skipped", skippedRunsText));
		
		// also add datamat normalization and window size info (for event-related data) to session file 
		// (for completeness, since all other datamat setup info is already being stored there)
		int volMeanNorm = 0;
		if (sessionFrame.normalizeVolumeCheckBox.isSelected()) {
			volMeanNorm = 1;
		}
		int refScanNorm = 1;
		if (!sessionFrame.normalizeScansCheckBox.isSelected()) {
			refScanNorm = 0;
		}
		int singSubjAnalysis = 0;
		if (sessionFrame.singleSubjectCheckBox.isSelected()) {
			singSubjAnalysis = 1;
		}
		sessionInfo.setField("normalize_volume_mean", new MLDouble("normalize_volume_mean",
				new double[][] {{volMeanNorm}}));
		sessionInfo.setField("normalize_with_baseline", new MLDouble("normalize_with_baseline",
				new double[][] {{refScanNorm}}));
		sessionInfo.setField("single_subject_analysis", new MLDouble("single_subject_analysis",
				new double[][] {{singSubjAnalysis}}));
		
		// window size for block PLS is always 1 so no need to save it
		if (!sessionFrame.isBlockedFmri) {
			try {
				int windowSz = Integer.parseInt(sessionFrame.windowField.getText());
				sessionInfo.setField("temporal_window_size", new MLDouble("temporal_window_size",
						new double[][] {{windowSz}}));
			} 
			catch (NumberFormatException nfe) {
				JOptionPane.showMessageDialog(null, "Must enter temporal window size.", 
						"Error", JOptionPane.ERROR_MESSAGE);
				return;
			}
			
				
		}
				
		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(sessionInfo);
        list.add(new MLChar("create_ver", "999999"));
        try {
        	new MatFileWriter(fileName, list);
        } catch(Exception ex) {
        	String message = "Could not save to session file " + 
			fileName + ".";
        
        	JOptionPane.showMessageDialog(null, message, "Error", JOptionPane.ERROR_MESSAGE);
        	throw new MissingResourceException(message,null,null);
        }
        
        if(stopDatamatGeneration){
        	throw new MissingResourceException("Image or mask datapath incorrect", 
        			null,null);
        }
	}
		
	/**
	 * Disable editing the filename text field in the JFileChooser dialog
	 * and set it to the datamat prefix.
	 * @param c (A JFileChooser upcast as a Container).
	 */
	private void disableText(Container c,String datamatPrefix){
		
		for(Component comp : c.getComponents()){
			
			if (comp instanceof JPanel){
				disableText((JPanel)comp,datamatPrefix);
			}
			else if(comp instanceof JTextField){
				((JTextField) comp).setText(datamatPrefix);
				((JTextField) comp).setEditable(false);
			}
		}
	}
	
	/**
	 * 
	 * @return Return the session file name without its prefix.
	 */
	String prefixWithoutSuffix(){
		int cutoff;
		if(sessionFrame.isBlockedFmri){ //PLS block
			cutoff = fileName.indexOf(BfMRISessionFileFilter.EXTENSION);
		}
		else{ //NPairs event and PLS event.
			cutoff = fileName.indexOf(fMRISessionFileFilter.EXTENSION);
		}
		
		if(cutoff == -1){ // Prefix has already been fixed.
			return fileName;
		}
		return fileName.substring(0, cutoff);
	}
	
	/**
	 * Update the session file prefix so it agrees with the datamat prefix.
	 * @throws MissingResourceException if the user has cancelled the saveAs
	 * dialog which means that session profile and datamat prefix are still
	 * out of sync.
	 */
	private void checkFilePath(){
		
		//Just get the actual name of the to be file, stripped of the 
		//absolute path prefix.
		String prefix = new File(prefixWithoutSuffix()).getName();
		String datamatPrefix = sessionFrame.datamatPrefixField.getText();
		
		if(!prefix.equals(datamatPrefix)){
			
			//The datamat prefix has changed so the user must now save
			//the session file under the same prefix. open a prompt
			//allowing them to do this.
			saveAs();
		}
	}
	
	
}

final class RunGenerateDatamatActionListener implements ActionListener {
	
	private SessionProfileFrame sessionFrame = null;
	
	private JTextField datamatPrefixField = null;
	
	private JTextField brainRegionFileField = null;
	
	private JTextField brainRegionThresholdField = null;
	
	private JTextField scansField = null;
	
	private JTextField runsField = null;
	
	private JTextField slicesField = null;
	
	private JTextField windowField = null;
	
	private JCheckBox normalizeVolumeCheckBox = null;
	
	private JCheckBox normalizeScansCheckBox = null;
	
	private JCheckBox singleSubjectCheckBox = null;
	
	private JRadioButton[] radioButtons = null;
	
	public RunGenerateDatamatActionListener(SessionProfileFrame sessionFrame) {
		this.sessionFrame = sessionFrame;
		this.brainRegionFileField = sessionFrame.brainRegionFileField;
		this.brainRegionThresholdField = sessionFrame.brainRegionThresholdField;
		this.scansField = sessionFrame.scansField;
		this.runsField = sessionFrame.runsField;
		this.slicesField = sessionFrame.slicesField;
		this.windowField = sessionFrame.windowField;
		this.normalizeVolumeCheckBox = sessionFrame.normalizeVolumeCheckBox;
		this.normalizeScansCheckBox = sessionFrame.normalizeScansCheckBox;
		this.singleSubjectCheckBox = sessionFrame.singleSubjectCheckBox;
		this.radioButtons = sessionFrame.brainRegionRadioButtons;
		this.datamatPrefixField = sessionFrame.datamatPrefixField;
	}
	
	
	public void actionPerformed(ActionEvent ae) {
		
		try{
			SessionProfileMenuBar menubar; 
			menubar = ((SessionProfileMenuBar)sessionFrame.getJMenuBar());
			// set invisible windowField to 1 for block fMRI (can't be set
			// in GUI since invisible)
			if (sessionFrame.isBlockedFmri) {
				sessionFrame.windowField.setText("1");
			}
			
			if(sessionFrame.getSessionFilename() != null){
				
				String prefix = new File(menubar.prefixWithoutSuffix()).getName();
				String datamatPrefix = sessionFrame.datamatPrefixField.getText();
				
				if(prefix.equals(datamatPrefix) && 
				   new File(sessionFrame.getSessionFilename()).exists()){
					
					int result = JOptionPane.showConfirmDialog(sessionFrame,
							"Saving session file " 
							+ sessionFrame.getSessionFilename() + ".",
							"Overwrite?",JOptionPane.OK_CANCEL_OPTION);
					if(result == JOptionPane.OK_OPTION){
						menubar.save();
					}else{
						return;
					}
				}else{
					menubar.saveAs();
				}
			}else{
				menubar.saveAs();
			}
			
		}catch(MissingResourceException e){
			return; //Error while saving, abort datamat generation.
		}
		
		
//		if(sessionFrame.usingDatamats()){
//			if(!checkSavePath()) return;
//		}
		
		String sessionFile = sessionFrame.getSessionFilename();
				
		Vector<String[]> conditionInfo = sessionFrame.conditionInfo;
		Vector<RunInformation> runInfo = sessionFrame.runInfo;
		String[] sIgnoreRuns = runsField.getText().split(" ");
		int[] ignoreRuns = null;
		
		if(sIgnoreRuns[0].length() != 0) {
			ignoreRuns = new int[sIgnoreRuns.length];
			for(int i = 0; i < sIgnoreRuns.length; i++) {
				ignoreRuns[i] = Integer.parseInt(sIgnoreRuns[i]);
			}
		}
		
		boolean useBrainMask = radioButtons[1].isSelected();	
		
		boolean useAllVox = radioButtons[2].isSelected();
		String brainMaskFile = brainRegionFileField.getText();
		double coordThresh = Double.parseDouble(brainRegionThresholdField.getText());
		
		if(useBrainMask) {
			// TODO: check if anything needs to be added here
		}

		String[] sIgnoreSlices = slicesField.getText().split(" ");
		int[] ignoreSlices = null;
		if(sIgnoreSlices[0].length() != 0) {
			ignoreSlices = new int[sIgnoreSlices.length];
			for(int i = 0; i < sIgnoreSlices.length; i++) {
				ignoreSlices[i] = Integer.parseInt(sIgnoreSlices[i]);
			}
		}
		
		boolean normalizeMeanVolume = normalizeVolumeCheckBox.isSelected();
		int numSkippedScans = Integer.parseInt(scansField.getText());
		int windowSize = Integer.parseInt(windowField.getText());
		boolean mergeAcrossRunsFlag = sessionFrame.mergeAcrossRuns();
		boolean normalizeSignalMean = normalizeScansCheckBox.isSelected();
		boolean considerAllVoxels = useAllVox;
		boolean singleSubject = singleSubjectCheckBox.isSelected();
		String datamatPrefix = datamatPrefixField.getText();
		
		try {
			StreamedProgressDialog dialog = new StreamedProgressDialog(sessionFrame, runInfo.size() * conditionInfo.size() + 1);
			PipedOutputStream pos = new PipedOutputStream();
			dialog.connectWriter(pos);
			StreamedProgressHelper helper = new StreamedProgressHelper();
			helper.addStream(pos);
			
			RunGenerateDatamat worker = new RunGenerateDatamat(sessionFrame.isBlockedFmri, 
					ignoreRuns, sessionFile, useBrainMask, brainMaskFile, coordThresh, 
					ignoreSlices, normalizeMeanVolume, numSkippedScans, windowSize, 
					mergeAcrossRunsFlag, normalizeSignalMean, considerAllVoxels, 
					singleSubject, conditionInfo, runInfo, datamatPrefix);
			
			dialog.worker = worker;
			worker.progress = helper;
			worker.start();
			sessionFrame.setDatamatGenerator(worker);
			
		} catch (IOException ioex) {
			ioex.printStackTrace();
		}
				
//		new ProgressDialog(sessionFrame, runInfo.size() * conditionInfo.size() + 1, 
//				worker);
	}
	
	
}

final class ChangeDataPathsListener implements ActionListener {
	
	private SessionProfileFrame sessionFrame = null;
	
	public ChangeDataPathsListener(SessionProfileFrame sessionFrame) {
		this.sessionFrame = sessionFrame;
	}
	
//	public ChangeDataPathsListener(NpairsSessionProfileFrame nSessionFrame) {
//		this.nSessionFrame = nSessionFrame;
//	}
	public void actionPerformed(ActionEvent e) {
//		if (sessionFrame != null) {
			new ChangeDataPathsFrame(sessionFrame);
//		}
//		else {
//			new ChangeDataPathsFrame(nSessionFrame);
//		}
	}
}

