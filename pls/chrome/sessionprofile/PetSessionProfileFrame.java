package pls.chrome.sessionprofile;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import java.util.ArrayList;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Iterator;

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
import pls.sessionprofile.PetSubjectGenerateDatamat;
import pls.sessionprofile.RunInformation;
import pls.sessionprofile.PetRunInformation;
import pls.sessionprofile.PetSubjectInformation;


import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.ProgressDialog;
import pls.shared.MLFuncs;
import pls.shared.AnalyzeNiftiFileFilter;
import pls.shared.fMRISessionFileFilter;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.PetSessionFileFilter;

public class PetSessionProfileFrame extends JFrame {
    protected int subjectInitial;
	
    private PetSessionProfileFrame frame = null;
    
    private PetSessionProfileMenuBar menubar = null;
	
    private JLabel numConditionsLabel = new JLabel("Number of Conditions: 0");
    
    private JLabel numSubjectLabel = new JLabel("Number of Subjects: 0");

    private BrainViewerPanel brainPanel = new BrainViewerPanel();
	
	protected JTextField descriptionField = new JTextField();
	
	protected JTextField datamatPrefixField = new JTextField();
    
	protected ButtonGroup mergeDataGroup = new ButtonGroup();
    
	// {Condition Name, Reference Scan Onset, Number of Reference Scans}
	protected Vector<String[]> conditionInfo = new Vector<String[]>();
    
	protected Vector<PetSubjectInformation> subjectInfo = new Vector<PetSubjectInformation>();
	
	protected Vector<PetRunInformation> runInfo = new Vector<PetRunInformation>();
	
    protected JTextField brainRegionFileField = new JTextField();
	
    protected JTextField brainRegionThresholdField = new JTextField("0.25");
	
    protected JTextField scansField = new JTextField("0");
	
    protected JTextField runsField = new JTextField();
	
    protected JTextField slicesField = new JTextField();
	
    protected JTextField windowField = new JTextField("8");

    protected JCheckBox considerAllVoxel = new JCheckBox("Consider all voxels as brain");
    
    protected JCheckBox normalizeVolumeCheckBox = new JCheckBox("Normalize data with volume mean");
	
    protected JCheckBox normalizeScansCheckBox = new JCheckBox("Normalize data with reference scans");
	
    protected JCheckBox singleSubjectCheckBox = new JCheckBox("Single Subject Analysis");
	
    protected JRadioButton[] brainRegionRadioButtons = new JRadioButton[2];
    
    //protected boolean isBlockedFmri = false;
	
    public String cwd="";
    
	public PetSessionProfileFrame() {
		super("Create new PET session information");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	    
	    frame = this;
	  //  this.isBlockedFmri = isBlockedFmri;
	    
	    
	    /*************************************************/
	    // Session File Panel
	    
	    subjectInitial = -1;
	    JLabel descriptionLabel = new JLabel("Session Description");
	    descriptionField.setColumns(12);

	    JLabel prefixLabel = new JLabel("<html><span style=\"color: red;\">*</span> Datamat Prefix</html>");
	    datamatPrefixField.setColumns(12);
	    
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 5, 5));
        labelPane.add(descriptionLabel);
        labelPane.add(prefixLabel);
        
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 5, 5));
        fieldPane.add(descriptionField);
        fieldPane.add(datamatPrefixField);

	    JPanel topPane = new JPanel(new GridLayout(0, 2));
        topPane.add(labelPane);
        topPane.add(fieldPane);
        
        JRadioButton[] mergeDataRadioButtons = new JRadioButton[2];
/*
        mergeDataRadioButtons[0] = new JRadioButton("Across All Runs");
        mergeDataRadioButtons[0].setSelected(true);
        mergeDataRadioButtons[1] = new JRadioButton("Within Each Run");
        
        mergeDataGroup.add(mergeDataRadioButtons[0]);
        mergeDataGroup.add(mergeDataRadioButtons[1]);

		JPanel radioBox = new JPanel(new GridLayout(1, 0, 100, 0));
        
	    
		radioBox.add(mergeDataRadioButtons[0]);
		radioBox.add(mergeDataRadioButtons[1]);
*/
        Border border = BorderFactory.createEtchedBorder();
	    //radioBox.setBorder(border);

	    
        JButton editConditionsButton = new JButton("Input Conditions", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Edit16.gif")));
        editConditionsButton.setIconTextGap(15);
        editConditionsButton.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) { new PetEditConditionsFrame(frame); }});
		
        JButton editRunsButton = new JButton("Select Subjects", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Edit16.gif")));
        editRunsButton.setIconTextGap(15);
        editRunsButton.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) { new PetEditSubjectFrame(frame); }});

        labelPane = new JPanel(new GridLayout(0, 1, 100, 0));
        labelPane.add(numConditionsLabel);
        labelPane.add(numSubjectLabel);
        
        fieldPane = new JPanel(new GridLayout(0, 1, 100, 0));
        fieldPane.add(editConditionsButton);
        fieldPane.add(editRunsButton);
        
	    JPanel runConditionPane = new JPanel(new GridLayout(0, 2));
	    runConditionPane.add(labelPane, BorderLayout.CENTER);
	    runConditionPane.add(fieldPane, BorderLayout.LINE_END);
        
	    JPanel sfPanel = new JPanel(new GridLayout(0, 1, 20, 20));
	    sfPanel.add(topPane);
	    
	    //sfPanel.add(radioBox);
	    sfPanel.add(runConditionPane);
	    border = BorderFactory.createEmptyBorder(50, 0, 0, 0);
	    sfPanel.setBorder(border);
	    JPanel sessionFilePanel = new JPanel();
		sessionFilePanel.add(sfPanel);
		
		/*************************************************/
		// Brain region panel
         
		brainRegionRadioButtons[0] = new JRadioButton("Define brain region automatically:");
		brainRegionRadioButtons[0].setSelected(true);
		brainRegionRadioButtons[1] = new JRadioButton("Use predefined brain region:");
		ButtonGroup group = new ButtonGroup();
		group.add(brainRegionRadioButtons[0]);
		group.add(brainRegionRadioButtons[1]);
		
		brainRegionFileField.setColumns(25);
		brainRegionFileField.setEnabled(false);
		final JButton browseButton = new JButton("Select mask file", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		browseButton.setIconTextGap(15);
		browseButton.setEnabled(false);
	    
		browseButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new AnalyzeNiftiFileFilter());
				int option = chooser.showDialog(PetSessionProfileFrame.this, "Select contrast data file");
				if (option == JFileChooser.APPROVE_OPTION) {
					brainRegionFileField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
		
		JPanel predefinedPanel = new JPanel();
		predefinedPanel.add(brainRegionFileField);
		predefinedPanel.add(browseButton);

		final JLabel thresholdLabel = new JLabel("Threshold");
		brainRegionThresholdField.setColumns(5);
		
		JPanel definedPanel = new JPanel();
		definedPanel.add(thresholdLabel);
		definedPanel.add(brainRegionThresholdField);
		
		ItemListener activateBehaviour = new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				if(brainRegionRadioButtons[1].isSelected()) {
					brainRegionFileField.setEnabled(true);
					browseButton.setEnabled(true);
					thresholdLabel.setEnabled(false);
					brainRegionThresholdField.setEnabled(false);
					considerAllVoxel.setEnabled(false);
					considerAllVoxel.setSelected(false);
					brainRegionThresholdField.setText("0.25");
				} else {
					brainRegionFileField.setEnabled(false);
					browseButton.setEnabled(false);
					thresholdLabel.setEnabled(true);
					brainRegionThresholdField.setEnabled(true);
					considerAllVoxel.setEnabled(true);
				}
			}
		};
		
		
		definedPanel.add(considerAllVoxel);
		
		considerAllVoxel.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				brainRegionThresholdField.setText("0");
			}
		});

		brainRegionRadioButtons[1].addItemListener(activateBehaviour);
		
		JPanel VolumeCheckBoxPanel = new JPanel(new GridLayout(0, 1));
		normalizeVolumeCheckBox.setSelected(true);
		VolumeCheckBoxPanel.add(normalizeVolumeCheckBox);
		
		
		normalizeVolumeCheckBox.addItemListener(new ItemListener()
		{
			public void itemStateChanged(ItemEvent e){
				if(normalizeVolumeCheckBox.isSelected()== false)
				JOptionPane.showMessageDialog(null, "un check is not recommended.", "Warning", JOptionPane.WARNING_MESSAGE);	
			}
		});
			
		JPanel brPanel = new JPanel(new GridLayout(0, 1));
		brPanel.add(brainRegionRadioButtons[0]);
		brPanel.add(definedPanel);
		brPanel.add(brainRegionRadioButtons[1]);
		brPanel.add(predefinedPanel);
		
		brPanel.add(VolumeCheckBoxPanel);
		
	    border = BorderFactory.createEmptyBorder(50, 0, 0, 0);
	    brPanel.setBorder(border);
		JPanel brainRegionPanel = new JPanel();
		brainRegionPanel.add(brPanel);
		
		
		/*************************************************/
		// Datamat panel
	/*	
		scansField.setColumns(10);
		runsField.setColumns(10);
		slicesField.setColumns(10);
		windowField.setColumns(10);
		
		normalizeScansCheckBox.setSelected(true);
	    
	    JPanel leftPanel = new JPanel(new GridLayout(0, 1, 5, 10));
	    leftPanel.add(new JLabel("Number of scans to be skipped"));
	    leftPanel.add(new JLabel("Runs to be ignored"));
	    leftPanel.add(new JLabel("Slices to be ignored"));
	    leftPanel.add(normalizeVolumeCheckBox);
	    leftPanel.add(normalizeScansCheckBox);
	    leftPanel.add(singleSubjectCheckBox);
        
        JPanel rightPanel = new JPanel(new GridLayout(0, 1, 5, 14));
        rightPanel.add(scansField);
        rightPanel.add(runsField);
        rightPanel.add(slicesField);
	
        rightPanel.add(new JLabel(""));
        rightPanel.add(new JLabel(""));
        rightPanel.add(new JLabel(""));
        
        JPanel leftRightPanel = new JPanel();
        leftRightPanel.add(leftPanel);
        leftRightPanel.add(rightPanel);
        
		JPanel datamatPanel = new JPanel(new GridLayout(0, 1));
	    datamatPanel.add(leftRightPanel);
	    border = BorderFactory.createEmptyBorder(50, 0, 0, 0);
	    datamatPanel.setBorder(border);
*/	    
	    // Buttons
		
        JButton clearSessionButton = new JButton("Clear Session", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Refresh16.gif")));
        clearSessionButton.setIconTextGap(15);
        clearSessionButton.addActionListener(new ActionListener(){
        	public void actionPerformed(ActionEvent e) {
        		descriptionField.setText(null);
        		datamatPrefixField.setText(null);
        		
        		//mergeDataGroup.getElements().nextElement().setSelected(true);
        		
        		conditionInfo.removeAllElements();
        		subjectInfo.removeAllElements();
        		
        		numConditionsLabel.setText("Number of Conditions: 0");
        		numSubjectLabel.setText("Number of Subjects: 0");
        		
        		brainRegionRadioButtons[0].setSelected(true);
        		brainRegionFileField.setText("");
        		brainRegionThresholdField.setText("0.15");
        		
        		considerAllVoxel.setEnabled(true);
				considerAllVoxel.setSelected(false);
			
        		/*
        		scansField.setText("0");
        		runsField.setText("");
        		slicesField.setText("");
        		windowField.setText("8");
        		normalizeVolumeCheckBox.setSelected(false);
        		normalizeScansCheckBox.setSelected(true);
        		singleSubjectCheckBox.setSelected(false);
        		*/
        		brainPanel.removeImages();
        	
        	}});
		
        JButton createStDatamatButton = new JButton("Create Datamat", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
        createStDatamatButton.setIconTextGap(15);
        createStDatamatButton.addActionListener(new PetSubjectGenerateDatamatActionListener(this));
        
		JPanel buttonPane = new JPanel(new GridLayout(1, 0));
        buttonPane.add(clearSessionButton);
        buttonPane.add(createStDatamatButton);
		
        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("Session File", sessionFilePanel);
        tabs.addTab("Brain Region", brainRegionPanel);
       // tabs.addTab("Spatial-Temporal Datamat", datamatPanel);
        tabs.addTab("Sample Image Orientation", brainPanel);
		
		JPanel mainPanel = new JPanel(new BorderLayout(5, 5));
		mainPanel.add(tabs, BorderLayout.NORTH);
		mainPanel.add(buttonPane, BorderLayout.SOUTH);
	    border = BorderFactory.createEtchedBorder();
	    border = BorderFactory.createEmptyBorder(5, 5, 5, 5);
	    mainPanel.setBorder(border);
		
        add(mainPanel);
	    
        menubar = new PetSessionProfileMenuBar(this);
	    setJMenuBar(menubar);
	    
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
	
	public String getSessionFilename() {
		return menubar.fileName;
	}
	
	public void updateConditions(Vector<String[]> newConditionInfo) {
		conditionInfo = newConditionInfo;
		numConditionsLabel.setText("Number of Conditions: " + conditionInfo.size());
	}

	public void updateSubjects(Vector<PetSubjectInformation> newSubjectInfo) {
		subjectInfo = newSubjectInfo;
		int numSubjects = subjectInfo.size();
		numSubjectLabel.setText("Number of Subjects: " + numSubjects);
		if(numSubjects > 0) {
			String dir = subjectInfo.get(0).dataDirectory;
			String subjectfile = subjectInfo.get(0).subjectFiles[0];
			try {
				brainPanel.loadImage(dir,subjectfile);
			} catch(Exception ex) {
				JOptionPane.showMessageDialog(null, "Data path seems to be incorrect.", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
	}
	public void BrainPanelLoadImage(Vector<PetSubjectInformation> newSubjectInfo){
		int numSubjects = subjectInfo.size();
		if(numSubjects > 0) {
			String dir = subjectInfo.get(0).dataDirectory;
			String subjectfile = subjectInfo.get(0).subjectFiles[0];
			try {
				brainPanel.loadImage(dir,subjectfile);
			} catch(Exception ex) {
				JOptionPane.showMessageDialog(null, "Data path seems to be incorrect.", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
	}
	public void updateRuns(Vector<PetRunInformation> newRunInfo) {
		runInfo = newRunInfo;
		int numRuns = runInfo.size();
		numSubjectLabel.setText("Number of Runs: " + numRuns);
		if(numRuns > 0) {
			String dir = runInfo.get(0).dataDirectory;
			String file = runInfo.get(0).dataFiles.split(" ")[0];
			try {
				brainPanel.loadImage(dir, file);
			} catch(Exception ex) {
				JOptionPane.showMessageDialog(null, "Data path seems to be incorrect.", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
	}
	
	public boolean isMergeAcrossRunsSelected() {
		return mergeDataGroup.getElements().nextElement().isSelected();
	}
	
	/**
	 * Get the run information from a matlab structure (from file presumably) into a Vector of our RunInformation class
	 * @param runs The run information in a matlab structure
	 * @param isBlockedFmri If it is blocked fmri or not
	 * @return The run information as a Vector of our RunInformation class
	 */
	
	public static Vector<PetSubjectInformation> getSubjectInformation(MLStructure subject) {
		Iterator i = subject.getAllFields().iterator();

		Vector<PetSubjectInformation> subjectInfo = new Vector<PetSubjectInformation>();
		
		while(i.hasNext()) {
			i.next();
			String dataDirectory = ((MLChar)i.next()).getString(0);
			MLCell cDataFiles = (MLCell)i.next();
			String dataFiles[] = new String[100];
			dataFiles[0]=new String("");
			for(int j = 0; j < cDataFiles.getM(); j++) {
				dataFiles[0] += ((MLChar)cDataFiles.get(j, 0)).getString(0) + " ";
			}
			dataFiles[0] = dataFiles[0].trim();
			
			i.next();
			MLCell cEventOnsets = (MLCell)i.next();
			ArrayList<String> eventOnsets = new ArrayList<String>();
			for(int j = 0; j < cEventOnsets.getN(); j++) {
				String onset = "";
				MLDouble dOnsets = (MLDouble)cEventOnsets.get(j);
				for(int k = 0; k < dOnsets.getM(); k++) {
					onset += dOnsets.get(k, 0).intValue() + " ";
				}
				eventOnsets.add(onset.trim());
			}
			
			PetSubjectInformation currSubjectInfo = new PetSubjectInformation(dataDirectory, dataFiles);
			
			subjectInfo.add(currSubjectInfo);
			
			//if(isBlockedFmri) {
				MLCell cEventLengths = (MLCell)i.next();
				ArrayList<String> eventLengths = new ArrayList<String>();
				for(int j = 0; j < cEventLengths.getN(); j++) {
					String length = "";
					MLDouble dLengths = (MLDouble)cEventLengths.get(j);
					for(int k = 0; k < dLengths.getM(); k++) {
						length += dLengths.get(k, 0).intValue() + " ";
					}
					eventLengths.add(length.trim());
				}
				//currSubjectInfo.lengths = eventLengths;
			//}
			
		}
		return subjectInfo;
	}

}

final class PetSessionProfileMenuBar extends BaseSaveMenuBar {
	
	private PetSessionProfileFrame sessionFrame = null;
	
	public PetSessionProfileMenuBar(PetSessionProfileFrame sessionFrame) {
		super(sessionFrame);
		
		this.sessionFrame = sessionFrame;

		setFileFilter(new PetSessionFileFilter(), "_PETsession.mat");
		
		//if(sessionFrame.isBlockedFmri) {
		//	setFileFilter(new BfMRISessionFileFilter(), BfMRISessionFileFilter.EXTENSION);
		/*} else {
			setFileFilter(new fMRISessionFileFilter(), fMRISessionFileFilter.EXTENSION);
		}
		*/
				
        // Build the edit menu
        JMenu editMenu = new JMenu("Edit");
        editMenu.setMnemonic('E');
        editMenu.getAccessibleContext().setAccessibleDescription("Edit session");
        add(editMenu, 1);
        
        JMenuItem changeDataPaths = new JMenuItem("Change Data Paths");
        changeDataPaths.setMnemonic('C');
        changeDataPaths.getAccessibleContext().setAccessibleDescription("Change the paths within the session files");
        changeDataPaths.addActionListener(new PetChangeDataPathsListener(sessionFrame));
        editMenu.add(changeDataPaths);
        
        JMenuItem mergeConditions = new JMenuItem("Merge Conditions");
        mergeConditions.setMnemonic('M');
        mergeConditions.getAccessibleContext().setAccessibleDescription("Merge the Conditions");
        mergeConditions.addActionListener(new PetMergeConditionsListener(sessionFrame));
        editMenu.add(mergeConditions);
	}
	
	public void load() {
					
		// Get needed variables from file
		MLStructure sessionInfo = null;
		
		try {
			sessionInfo = (MLStructure)new NewMatFileReader(fileName).getContent().get("session_info");			
		} catch(Exception ex) {
    		JOptionPane.showMessageDialog(null, "Session file " + fileName + " could not be loaded.", "Error", JOptionPane.ERROR_MESSAGE);
    		return;
		}
		
		sessionFrame.descriptionField.setText(((MLChar)sessionInfo.getField("description")).getString(0));
		
		sessionFrame.datamatPrefixField.setText(((MLChar)sessionInfo.getField("datamat_prefix")).getString(0));
	
		int numCondition = ((MLDouble)sessionInfo.getField("num_conditions")).get(0, 0).intValue();
		
		String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
		
		Vector<String[]> conditionInfo = new Vector<String[]>();
	    
		for(int i = 0; i < conditions.length; i++) {
	    	String conditionName = conditions[i];
	     	conditionInfo.add(new String[]{conditionName});
	    }
		sessionFrame.updateConditions(conditionInfo);
		
		Vector<PetSubjectInformation> subjectInfo = new Vector<PetSubjectInformation>();
		
		int numSubject = ((MLDouble)sessionInfo.getField("num_subjects")).get(0, 0).intValue();
		
		MLCell subject = new MLCell("subject",new int[]{1, numSubject});

	    String[] csubject = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("subject"));

	    String files="";
	//    System.out.println("condition number"+numCondition+" Numberofsubject "+numSubject);

	    MLCell subjfiles = (MLCell)sessionInfo.getField("subj_files",0);	
	    PetSubjectInformation psinfo;
	    
	    for(int i = 0; i < numSubject; i++) {
	    	files ="";
	    	for(int j=0; j<numCondition; j++)
	    	{
	    		files +=((MLChar)subjfiles.get(j, i)).getString(0) + " ";

	    	}
	        String [] subjectFiles = files.split(" ");
	        //for(int k=0; k<subjectFiles.length; k++)
	        //	System.out.println("subject "+i + " "+ subjectFiles[k]);
    		psinfo = new PetSubjectInformation(csubject[i],subjectFiles);
    		subjectInfo.addElement(psinfo);
	   }
	   sessionFrame.updateSubjects(subjectInfo);
	   
	   int sInitial = ((MLDouble)sessionInfo.getField("num_subj_init")).get(0, 0).intValue();
	   
	   //System.out.println("num_subject_initial"+sInitial);
	   
	   sessionFrame.subjectInitial = sInitial;
	}
	
	public void save(){
		MLStructure sessionInfo = new MLStructure("session_info", new int[]{1, 1});
		
		sessionInfo.setField("description", new MLChar("description", sessionFrame.descriptionField.getText()));		
		sessionInfo.setField("pls_data_path", new MLChar("pls_data_path", PetChangeDataPathsFrame.getCommonPath(sessionFrame.subjectInfo)));
		
		sessionInfo.setField("numbehavior", new MLDouble("numbehavior", new double[][]{{0}}));
		
		sessionInfo.setField("behavdata", new MLDouble("behavedata", new double[][]{{}}));
			    
		sessionInfo.setField("behavname",new MLCell("behavename", new int[]{0,0}));		
					
		sessionInfo.setField("datamat_prefix", new MLChar("datamat_prefix", sessionFrame.datamatPrefixField.getText()));
		
        int numConditions = sessionFrame.conditionInfo.size();
        int numSubject = sessionFrame.subjectInfo.size();
        
        sessionInfo.setField("num_conditions", new MLDouble("num_conditions", new double[][]{{numConditions}}));
                     
        MLCell conditions = new MLCell("condition", new int[]{1, numConditions});
       // MLCell conditionBaseline = new MLCell("condition_baseline", new int[]{1, numConditions});
        for(int i = 0; i < numConditions; i++) {
	    	String conditionName = sessionFrame.conditionInfo.get(i)[0];
	    	//double refScanOnset = new Double(sessionFrame.conditionInfo.get(i)[1]).doubleValue();
	    	//double numRefScans = new Double(sessionFrame.conditionInfo.get(i)[2]).doubleValue();
        	conditions.set(new MLChar("condition", conditionName), 0, i);
        	//conditionBaseline.set(new MLDouble("conditions_baseline" + i, new double[][]{{refScanOnset, numRefScans}}), 0, i);
        }
        
        sessionInfo.setField("condition", conditions);
        
      	//conditions = new MLCell("conditions", new int[]{1, numSubject * numConditions});
        int count = 0;
        	
        sessionInfo.setField("num_subjects", new MLDouble("num_subjects", new double[][]{{numSubject}}));        
	
        
		//MLStructure subject = new MLStructure("run", new int[]{1, numSubject});
        MLCell subject = new MLCell("subject",new int[]{1, numSubject});
		
        
        MLCell subjectName = new MLCell("subj_name",new int[]{1, numSubject});

    	for(int i = 0; i < numSubject; i++) {
    		subject.set(new MLChar("subject",sessionFrame.subjectInfo.get(i).dataDirectory),0,i);
    		String directory = sessionFrame.subjectInfo.get(i).dataDirectory;
    		System.out.println("directory*****"+directory+"***********");
    		int index = sessionFrame.subjectInfo.get(i).dataDirectory.lastIndexOf("\\");    		
    		String path =sessionFrame.subjectInfo.get(i).dataDirectory;
    		int len = path.length();
    		String last =sessionFrame.subjectInfo.get(i).dataDirectory.substring(index+1, len);
    		subjectName.set(new MLChar("subj_name",last),0,i);
    		
    	}
    	int subjectInitial = sessionFrame.subjectInitial;
        
    	sessionInfo.setField("subject", subject);
    	sessionInfo.setField("subj_name", subjectName);
    	
    	MLCell cFiles = new MLCell("subj_files", new int[]{numConditions,numSubject});
         
        for(int j=0; j<numConditions;j++){
        	for(int i = 0; i<numSubject; i++) {
        		String file = sessionFrame.subjectInfo.get(i).subjectFiles[j];
				cFiles.set(new MLChar("subj_files", file), j, i);				
        	}        	
        }
        
        sessionInfo.setField("subj_files", cFiles,0);
        
	    sessionInfo.setField("img_ext", new MLChar("img_ext","*.img"));			
        sessionInfo.setField("num_subj_init", new MLDouble("num_subj_init", new double[][]{{subjectInitial}}));            	        	        
        ArrayList<MLArray> list = new ArrayList<MLArray>();
        list.add(new MLChar("create_ver", "999999"));
        list.add(sessionInfo);
        
        System.out.println("filename****"+fileName+"****");
        
        try {
        	new MatFileWriter(fileName, list);
        	
        } catch(Exception ex) {
        	JOptionPane.showMessageDialog(null, "Could not save to session file " + fileName + ".", "Error", JOptionPane.ERROR_MESSAGE);
        	return;
        }
	}
}

final class PetSubjectGenerateDatamatActionListener implements ActionListener {
	
	private PetSessionProfileFrame sessionFrame = null;
	
	private JTextField datamatPrefixField = null;
	
	private JTextField brainRegionFileField = null;
	
	private JTextField brainRegionThresholdField = null;
	
	private JRadioButton[] radioButtons = null;
	
	private JCheckBox normalizeVolumeCheckBox = null;
	
	/*
	private JTextField scansField = null;
	
	private JTextField runsField = null;
	
	private JTextField slicesField = null;
	
	private JTextField windowField = null;
	
	
	
	private JCheckBox normalizeScansCheckBox = null;
	
	private JCheckBox singleSubjectCheckBox = null;
	*/
	
	public PetSubjectGenerateDatamatActionListener(PetSessionProfileFrame sessionFrame) {
		this.sessionFrame = sessionFrame;
		this.datamatPrefixField = sessionFrame.datamatPrefixField;		
		this.brainRegionFileField = sessionFrame.brainRegionFileField;
		this.brainRegionThresholdField = sessionFrame.brainRegionThresholdField;
		this.radioButtons = sessionFrame.brainRegionRadioButtons;
		this.normalizeVolumeCheckBox = sessionFrame.normalizeVolumeCheckBox;
		
		/*
		this.scansField = sessionFrame.scansField;
		this.runsField = sessionFrame.runsField;
		this.slicesField = sessionFrame.slicesField;
		this.windowField = sessionFrame.windowField;
		
		this.normalizeScansCheckBox = sessionFrame.normalizeScansCheckBox;
		this.singleSubjectCheckBox = sessionFrame.singleSubjectCheckBox;		
		*/
	}
	
	
	public void actionPerformed(ActionEvent ae) {
		Vector<String[]> conditionInfo = sessionFrame.conditionInfo;
		Vector<PetSubjectInformation> subjectInfo = sessionFrame.subjectInfo;
	    int subjectInitial = sessionFrame.subjectInitial;
		String sessionFile = sessionFrame.getSessionFilename();
				
		boolean useBrainMask = radioButtons[1].isSelected();
		String brainMaskFile = brainRegionFileField.getText();
		double coordThresh = Double.parseDouble(brainRegionThresholdField.getText());
		
		String datamatPrefix = datamatPrefixField.getText();
		
		boolean normalizeMeanVolume = normalizeVolumeCheckBox.isSelected();
		
		boolean mergedConditions = false;
		String [] mergedconditionList = new String[2];

		
		/*
		String[] sIgnoreSlices = slicesField.getText().split(" ");
		int[] ignoreSlices = null;
		if(sIgnoreSlices[0].length() != 0) {
			ignoreSlices = new int[sIgnoreSlices.length];
			for(int i = 0; i < sIgnoreSlices.length; i++) {
				ignoreSlices[i] = Integer.parseInt(sIgnoreSlices[i]);
			}
		}		
		int numSkippedScans = Integer.parseInt(scansField.getText());
		int windowSize = Integer.parseInt(windowField.getText());
		boolean mergeAcrossRunsFlag = sessionFrame.isMergeAcrossRunsSelected();
		boolean normalizeSignalMean = normalizeScansCheckBox.isSelected();		
		boolean singleSubject = singleSubjectCheckBox.isSelected();
		*/
		
    	//RunGenerateDatamat worker = new RunGenerateDatamat(sessionFrame.isBlockedFmri, ignoreRuns, sessionFile, useBrainMask, brainMaskFile, coordThresh, ignoreSlices, normalizeMeanVolume, numSkippedScans, windowSize, mergeAcrossRunsFlag, normalizeSignalMean, considerAllVoxels, singleSubject, conditionInfo, runInfo, datamatPrefix);		
		PetSubjectGenerateDatamat worker = new PetSubjectGenerateDatamat(sessionFile, useBrainMask, brainMaskFile, coordThresh, normalizeMeanVolume,conditionInfo, subjectInfo, datamatPrefix,mergedConditions,mergedconditionList);
		new ProgressDialog(sessionFrame, subjectInfo.size() * conditionInfo.size() + 1, worker);
	}
}

final class PetChangeDataPathsListener implements ActionListener {
	
	private PetSessionProfileFrame sessionFrame = null;
	
	public PetChangeDataPathsListener(PetSessionProfileFrame sessionFrame) {
		this.sessionFrame = sessionFrame;
	}
	
	public void actionPerformed(ActionEvent e) {
		new PetChangeDataPathsFrame(sessionFrame);
	}
}
final class PetMergeConditionsListener implements ActionListener {
	
	private PetSessionProfileFrame sessionFrame = null;
	
	public PetMergeConditionsListener(PetSessionProfileFrame sessionFrame) {
		this.sessionFrame = sessionFrame;
	}
	
	public void actionPerformed(ActionEvent e) {
		new PetMergeConditionsFrame(sessionFrame);
	}
}
