package pls.chrome.analysis;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.filechooser.FileFilter;
import javax.swing.table.DefaultTableModel;

import npairs.NpairsjSetupParams;

import org.jfree.util.ArrayUtilities;

import pls.analysis.NpairsAnalysis;
import pls.sessionprofile.SessionProfile;
import pls.shared.MLFuncs;
import pls.shared.NpairsBlockSessionFileFilter;
import pls.shared.NpairsClassFileFilter;
import pls.shared.NpairsEVDFileFilter;
import pls.shared.NpairsSplitsInfoFileFilter;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.StreamedProgressDialog;
import pls.shared.StreamedProgressHelper;
import pls.shared.fMRISessionFileFilter;

import com.jmatio.types.MLCell;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

@SuppressWarnings("serial")
public class NpairsAnalysisFrame extends JFrame {

	NpairsAnalysisMenuBar menuBar;
	
	final NpairsSessionProfilePanel npairsSessionProfilePanel;
	public final ResamplingPanel resamplingOptionsPanel;
	ModelOptionsPanel npairsAnalysisTypeOptionsPanel;
	final DataReductionOptionsPanel dataReducOptsPanel;

	final JTextField resultsFilePrefField;	
	public static final String DEFAULT_RESULT_FILE_PREF = "npairsResult/npairs";
		
	String matlibType = pls.shared.GlobalVariablesFunctions.matrixLibrary;
	String matlibTypeForInitFeatSel = pls.shared.GlobalVariablesFunctions.matrixLibrary;
	
	boolean saveLotsOfFiles = false;
	boolean saveSplitFiles = false;
	boolean rotatePCsToImgSpace = false;
	boolean saveDataPostPCA = false;
	private boolean classSelectionUpdated = true;
	
	protected boolean isBlocked;

	protected boolean calcProcWithEigims = false;
	
	public NpairsAnalysisFrame(boolean isBlocked) {
		super("NPAIRS Analysis");
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		final NpairsAnalysisFrame thisFrame = this;
		this.isBlocked = isBlocked;
		npairsSessionProfilePanel = new NpairsSessionProfilePanel(thisFrame);
		resamplingOptionsPanel = new ResamplingPanel(thisFrame);
		npairsAnalysisTypeOptionsPanel = new ModelOptionsPanel(thisFrame);
		dataReducOptsPanel = new DataReductionOptionsPanel();


		JPanel containerPane = new JPanel();
		containerPane.setLayout(new BoxLayout(containerPane, BoxLayout.Y_AXIS));
		containerPane.add(npairsSessionProfilePanel);
		containerPane.add(dataReducOptsPanel);
		containerPane.add(resamplingOptionsPanel);
		containerPane.add(npairsAnalysisTypeOptionsPanel);
		
		

		resultsFilePrefField = new JTextField(DEFAULT_RESULT_FILE_PREF);
		resultsFilePrefField.setColumns(12);
		JLabel resultsFilenameLabel = new JLabel("Results file prefix");
		JButton resultsButton = new JButton("Browse", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
		resultsButton.setIconTextGap(15);
		
		resultsButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				chooser.setFileFilter(new NpairsfMRIResultFileFilter());
				int option = chooser.showDialog(NpairsAnalysisFrame.this, "Save results to");
				if(option == JFileChooser.APPROVE_OPTION) {
					String selectedFile = chooser.getSelectedFile().getAbsolutePath();
					if (selectedFile.endsWith(NpairsfMRIResultFileFilter.EXTENSION)) {
						selectedFile = selectedFile.substring(0, selectedFile.indexOf(
								NpairsfMRIResultFileFilter.EXTENSION));
					}
					resultsFilePrefField.setText(selectedFile);
				}
			}
		});

		JPanel resultsPane = new JPanel(new GridLayout(1, 4, 8, 0));
		resultsPane.add(resultsFilePrefField);
		resultsPane.add(resultsFilenameLabel);
		resultsPane.add(resultsButton);
		resultsPane.setBorder(BorderFactory.createEmptyBorder(5,0,5,0));

		JPanel saveAndRunPane = new JPanel(new GridLayout(1, 0, 10, 0));
		final JButton runButton = new JButton("Run", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Play16.gif")));
		runButton.setIconTextGap(15);
		runButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				
				//1.save out temp setup file (which checks fields in the process)
				//2.pass temp file name to RunNpairs as setup file name.
				//3.analysis is run on temp file 
				//4.setup information is then saved in result dir.
				
				String error = "Could not create necessary temp setup file. " +
				"Aborting.";

				File temp;
				try{temp = File.createTempFile("analysis", null);}
				catch(IOException e){
					JOptionPane.showMessageDialog(thisFrame, error,"Error", 
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				SaveNpairsSetupParamsActionListener saveNPAIRS = 
					new SaveNpairsSetupParamsActionListener(thisFrame, 
														temp.getAbsolutePath());
				
				//Saving failed, so abort running an analysis. One of the fields 
				//may have been incorrect.
				if(!saveNPAIRS.save()) return;	
				
				try{
					new RunNpairs(thisFrame, 
							temp.getAbsolutePath(), 
							matlibType, 
							matlibTypeForInitFeatSel);
				}catch(IOException e){
					JOptionPane.showMessageDialog(thisFrame, 
							"An I/O error occurred while running the analysis." +
							"Reason: " + e.getMessage() + "\n Also see the stack trace.", 
							"Error", 
							JOptionPane.ERROR_MESSAGE);
				}
			}
		});
		saveAndRunPane.add(runButton);
		
		JPanel mainPane = new JPanel(new BorderLayout());
		mainPane.add(containerPane, BorderLayout.NORTH);
		mainPane.add(resultsPane, BorderLayout.CENTER);
		mainPane.add(saveAndRunPane, BorderLayout.SOUTH);
		mainPane.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
		JScrollPane mainScrollPane = new JScrollPane(mainPane);
		//mainScrollPane.add(mainPane);
		add(mainScrollPane);
		
		menuBar = new NpairsAnalysisMenuBar(this, null, null, isBlocked);
	    setJMenuBar(menuBar);
	    
		// Display the window
		pack();
		
		//adjusts frame's dimensions
		int verMagicNum = 75; 
		int horMagicNum = 50;
		setSize(getWidth() + horMagicNum, getHeight() - verMagicNum);
		
		// Position the frame on the screen
		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		int x = (int)(screen.getWidth() - getWidth()) / 2;
		int y = (int)(screen.getHeight() - getHeight()) / 2;
		setLocation(x, y);

		//setResizable(false);
		setVisible(true);
	}
	
	public String[] getSplitObjTypes() {
		return resamplingOptionsPanel.splitObjTypes;
	}
	
	/**
	 * Return the session profiles panel.
	 * @return The session profiles panel. The panel that contains the
	 * Add Group, Edit Group, and Delete Group buttons. 
	 */
	public NpairsSessionProfilePanel getSessionPanel(){
		return npairsSessionProfilePanel;
	}
	
	/**
	 * Do not call this function. It is exposed only for JUnit testing.
	 * @return The loaded session files.
	 */
	public Vector<String[]> getSessionFiles(){
		return npairsSessionProfilePanel.sessionProfiles;
	}
	
	/**
	 * Do not call this function. It is exposed only for JUnit testing.
	 * @return True if there was no conflict found among the conditions in the
	 * loaded session files (all files have the same number of conditions and
	 * the same conditions). False otherwise.
	 */
	public boolean updateSessions(){
		//Session files have been updated. Reflect this in the gui.
		
		//Check that the conditions are valid.
		if(!menuBar.updateDeselectMenuConds(getSessionFiles())){
			return false;
		}
		
		resamplingOptionsPanel.updateSplitPartitions();
		
		//update the session panel to show which groups are loaded.
		Vector<String []> copy = new Vector<String[]>(
				npairsSessionProfilePanel.sessionProfiles);
		
		npairsSessionProfilePanel.setSessProfiles(copy);
		return true;
	}
	
	public static String[] getConditionNames(String sessionFileName) throws FileNotFoundException, IOException {
		MLStructure sessProfStruct = (MLStructure)new NewMatFileReader(sessionFileName).
			getContent().get("session_info");
		String[] classNames = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition0"));
		return classNames;
	}
	
	/**
	 * Parse condition names from session files. Three conditions must hold, 1)
	 * all session files must have the same number of conditions and 2) all 
	 * session files must have the same conditions. 3) The conditions must be
	 * in the same order.
	 * @param groups groups of session files.
	 * @return list of conditions. null if an error occurred.
	 */
	public static String[] getConditionNamesFromGroups(Vector<String []> groups){
		if(groups.size() == 0){ 
			throw new IllegalArgumentException("Attempting to get " +
					"condition names from an empty set of groups");
		}
		
		//Set object used to determine if a condition does not exist in the
		//set of valid conditions. 
		Set<String> conditions = null;
		
		//Each session file must have their conditions listed in the same
		//order as every other session file.
		String[] conditionOrder = null;
		
		for(String[] group : groups){
			for(String session : group){
				String[] sessionConditions;
				try{
				
					sessionConditions = getConditionNames(session);
				
				}catch (FileNotFoundException e) {
					JOptionPane.showMessageDialog(null, 
							"Could not find session file "
							+ session+ ".\nPlease click on Edit Groups to update" +
							" file path if session files have been moved. " +
							"\nRemember to save the updated Analysis Setup "
							+ "file before running analysis.", 
							"Warning", JOptionPane.WARNING_MESSAGE);
					return null;
				}
				catch (IOException e) {
					JOptionPane.showMessageDialog(null, 
							"Could not load sessionfile " + session + 
							".", "Error", JOptionPane.ERROR_MESSAGE);
					return null;
				}
				
				if(conditions == null){
					conditions = new HashSet<String>(sessionConditions.length); 
					conditions.addAll(Arrays.asList(sessionConditions));	
					conditionOrder = sessionConditions;
				}
				//This session must have the same number of conditions
				//as all previous conditions.
				if(sessionConditions.length != conditions.size()){
					JOptionPane.showMessageDialog(null, 
							"All session files must have the same number" +
							" of conditions.\n " + session 
							,"Error", 
							JOptionPane.WARNING_MESSAGE);
					return null;
				}
				
				//This session must have the same conditions as well.
				for(String condition : sessionConditions){
					if(!conditions.contains(condition)){
						JOptionPane.showMessageDialog(null,
								" All files must have the same " +
								 "conditions.\n" + 
								session + " has an extra condition ("+condition
								 +").","Error", 
								JOptionPane.WARNING_MESSAGE);
						return null;
					}
				}
				
				//This session must have the same ordering of conditions.
				for(int i = 0; i < conditionOrder.length; i++){
					if(!conditionOrder[i].equals(sessionConditions[i])){
						JOptionPane.showMessageDialog(null, 
								"Conditions listed in a different order than"
								+ " previous session files \n" + session
								,"Error", 
								JOptionPane.WARNING_MESSAGE);
						return null;
					}
				}
			}
		}
		
		return conditionOrder;
				
		/* Dont sort the conditions.
		 String[] sortedConditions = new String[conditions.size()];
		conditions.toArray(sortedConditions);
		Arrays.sort(sortedConditions);
		return sortedConditions;*/
	}
	protected void setClassSelectUpdated(boolean state) {
		classSelectionUpdated = state;
	}
	
	protected boolean classSelectionUpdated() {
		return classSelectionUpdated;
	}

	// update number of total scans in current analysis
	public void updateNumScans() {
		// based 
		
	}
	
}


final class RunNpairs {

	public RunNpairs(JFrame parent, String npairsSetupParamsMatFileName, String matlibType, 
			String matlibTypeForInitFeatSel) throws IOException {
		
		StreamedProgressDialog dialog = new StreamedProgressDialog(parent, 100);
		StreamedProgressHelper helper = new StreamedProgressHelper();
		PipedOutputStream pos = new PipedOutputStream();
		
		dialog.connectWriter(pos);
		helper.addStream(pos);
		
		NpairsAnalysis worker = new NpairsAnalysis(npairsSetupParamsMatFileName, matlibType, 
				matlibTypeForInitFeatSel, ((NpairsAnalysisFrame)parent).isBlocked);
		
		dialog.worker = worker;
		
		worker.progress = helper;
		worker.start();
		
	}
}


@SuppressWarnings("serial")
final class NpairsSessionProfilePanel extends JPanel {

	// sessionProfiles contains one String array per group 
	// (of sessionfiles); each group's String array contains
	// full pathnames of sessionfiles 
	Vector<String[]> sessionProfiles = new Vector<String[]>();

	Vector<Integer> conditionSelection = new Vector<Integer>();

	final NpairsAnalysisFrame npairsFrame;
	
	private final SelectSessionFilesDialog selectDialog;
	
	final DefaultTableModel model;
	
	public NpairsSessionProfilePanel(NpairsAnalysisFrame parent) {
		
		npairsFrame = parent;
		
		JPanel mainPanel = new JPanel(new BorderLayout());

		model = new DefaultTableModel();
		final JTable jt = new JTable(model) {
			public boolean isCellEditable(int rowIndex, int vColIndex) {
				return false;
			}
		};
		jt.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		model.addColumn("Group");
		model.addColumn("Session Files");
		jt.getColumnModel().getColumn(0).setMaxWidth(50);
		jt.getColumnModel().getColumn(0).setResizable(false);
		jt.getColumnModel().getColumn(1).setResizable(false);
		jt.setPreferredScrollableViewportSize(new Dimension(300, 50));
			
		mainPanel.add(new JScrollPane(jt), BorderLayout.NORTH);
		
		FileFilter fileFilter = new NpairsBlockSessionFileFilter();
		if (!npairsFrame.isBlocked) {
			fileFilter = new fMRISessionFileFilter();
		}
		selectDialog = new SelectSessionFilesDialog(parent, fileFilter, 
				sessionProfiles, jt, model);
		selectDialog.setSize(640, 480);
        selectDialog.setResizable(false);
        
		JPanel buttonPanel = new JPanel();
		final JButton addButton = new JButton("Add Group", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		buttonPanel.add(addButton);
		final JButton editButton = new JButton("Edit Group", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Edit16.gif")));
		buttonPanel.add(editButton);
		editButton.setEnabled(false);
		final JButton deleteButton = new JButton("Delete Group", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Delete16.gif")));
		deleteButton.setEnabled(false);
		buttonPanel.add(deleteButton);
		
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);

		jt.addFocusListener(new FocusListener() {
			public void focusLost(FocusEvent ae) {
			}
			public void focusGained(FocusEvent ae) {
				if(jt.getRowCount() > 0) {
					deleteButton.setEnabled(true);
					editButton.setEnabled(true);
				}
			}
		});
		addButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				selectDialog.addNewGroup();
				
			}
		});
		editButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				int group = jt.getSelectedRow();
				if (group != -1) {
					selectDialog.editGroup(group);
					npairsFrame.setClassSelectUpdated(false);
				}
			}
		});
		deleteButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				int decr = 0;
				for(int r: jt.getSelectedRows()) {
					model.removeRow(r - decr);
					sessionProfiles.remove(r - decr);
					decr++;
				}
				for(int i = 0; i < jt.getRowCount(); i++) {
					jt.setValueAt(i + 1, i, 0);
				}
				if(jt.getRowCount() == 0) {
					deleteButton.setEnabled(false);
					editButton.setEnabled(false);
					//conditionSelection.removeAllElements();
					npairsFrame.menuBar.updateDeselectMenuConds(npairsFrame, null, null);
					npairsFrame.resamplingOptionsPanel.updateSplitPartitions();
				}
				else {
					npairsFrame.menuBar.updateDeselectMenuConds(sessionProfiles);
					npairsFrame.resamplingOptionsPanel.updateSplitPartitions();
					
				}
			}
		});

		mainPanel.setBorder(BorderFactory.createTitledBorder("Session Profiles"));
		mainPanel.setPreferredSize(new Dimension(500, 140));
		add(mainPanel);
	}
	
	public void setSessProfiles(Vector<String[]> sessionProfiles) {
		if (sessionProfiles != null) {
			model.setRowCount(0);
			this.sessionProfiles.removeAllElements();
			Iterator<String[]> iter = sessionProfiles.iterator();
			while (iter.hasNext()) {
				String fileList = "";
				String[] currGrpSessFiles = (String[])iter.next();
				for (String sf : currGrpSessFiles) {
					fileList += sf + "; ";
				}
				model.addRow(new Object[]{model.getRowCount() + 1, fileList});
				this.sessionProfiles.add(fileList.split("\\s*;\\s*"));
			}
		}
	}
	
	public void setSelectedClasses(Vector<Integer> conditionSelection) {
		this.conditionSelection = conditionSelection;
	}
	
	public SelectSessionFilesDialog getSelectDialog(){
		return selectDialog;
	}
}


@SuppressWarnings("serial")
final class ModelOptionsPanel extends JPanel {

	final NpairsAnalysisFrame npairsFrame;
	
	protected JCheckBox[] modelOptions = new JCheckBox[3]; // GLM, PCA,CVA
	
	protected JCheckBox[] preprocOptions = new JCheckBox[1]; // MSR

	protected JButton browseButt = new JButton("Browse", new ImageIcon(this.getClass().
			getResource("/toolbarButtonGraphics/general/Import16.gif")));
	
	protected JCheckBox normPCChkBox = new JCheckBox("Normalize PCs to have variance 1 before doing CVA");
	
	//private JCheckBox aCVABox = new JCheckBox("Do agnostic CVA (Sets unique class label for each input scan) ");
	
	protected JTextField cvaClassFilenameField  = new JTextField(20);
	
	protected JLabel cvaClassFilenameLabel = new JLabel("Load CVA class file?");

	public ModelOptionsPanel(NpairsAnalysisFrame frame) {

		/////////////////////RE-ENABLED FOR NOW///////////////
		cvaClassFilenameLabel.setEnabled(true);
		cvaClassFilenameField.setEnabled(true);
		browseButt.setEnabled(true);
		////////////////////////////////////////////////////
		
		this.npairsFrame = frame;
		
		JPanel mainModelOptsPane = new JPanel(new BorderLayout());
		
		JPanel preprocOptsPane = new JPanel(new GridLayout(1, preprocOptions.length));
		preprocOptsPane.setBorder(BorderFactory.createTitledBorder("Preprocessing"));
		preprocOptions[0] = new JCheckBox("MSR", true);
		for (int i =0; i < preprocOptions.length; ++i) {
			preprocOptsPane.add(preprocOptions[i]);
		}

		modelOptions[0] = new JCheckBox("GLM");
		modelOptions[0].setEnabled(false);
		modelOptions[1] = new JCheckBox("PCA", true);
		modelOptions[2] = new JCheckBox("CVA", true);

		// CVA
		modelOptions[2].addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				String splitObjType = (String)(npairsFrame.resamplingOptionsPanel.splitObjChoice.getSelectedItem());
				browseButt.setEnabled(modelOptions[2].isSelected() &&
						splitObjType.equals("Session (default)"));
				cvaClassFilenameLabel.setEnabled(modelOptions[2].isSelected() && 
						splitObjType.equals("Session (default)"));
				cvaClassFilenameField.setEnabled(modelOptions[2].isSelected() &&
						splitObjType.equals("Session (default)"));
				npairsFrame.dataReducOptsPanel.pcsInFullDataCVAField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.pcsInSplitCVAField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				normPCChkBox.setEnabled(modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.numPCsInFullDataCVALabel.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.numPCsInSplitCVALabel.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.fullDataMultFactField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.fullDataMultFactLab.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.setPCRange.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
		//		aCVABox.setEnabled(radioButtons[2].isSelected());
				
			}
		});
		
		// PCA
		modelOptions[1].addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				normPCChkBox.setEnabled(modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.pcsInFullDataCVAField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.pcsInSplitCVAField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.numPCsInFullDataCVALabel.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.numPCsInSplitCVALabel.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.fullDataMultFactField.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.fullDataMultFactLab.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
				npairsFrame.dataReducOptsPanel.setPCRange.setEnabled(
						modelOptions[2].isSelected() && modelOptions[1].isSelected());
			}
		});
		
//		aCVABox.setToolTipText("For each run, gives each input scan its own class label. " +
//			                   "Required: All input runs have the same number of scans.");
//		aCVABox.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent e) {
//				if (aCVABox.isSelected()) {
//				// TODO: if checkbox is selected, set cva class labels for 
//			    // agnostic CVA
//				Npairsj.output.println("You checked the box!");
//				}
//			}
//		});
		
		browseButt.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				chooser.setFileFilter(new NpairsClassFileFilter());
				int option = chooser.showDialog(ModelOptionsPanel.this, "Load CVA class labels file...");
				if(option == JFileChooser.APPROVE_OPTION) {
					if(!chooser.getSelectedFile().getName().contains(".")) {
						cvaClassFilenameField.setText(chooser.getSelectedFile().getAbsolutePath() + 
						".class");
					} else {
						cvaClassFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());					
					}

					// do everything cvaClassFilenameField ActionListener does 
					String classFilename = cvaClassFilenameField.getText().toString();

					try {		
						String[] newClassNames = NpairsAnalysisMenuBar.getClassNames(classFilename);
						// select all classes by default
						Vector<Integer> classSelection = new Vector<Integer>(newClassNames.length);
						for (int i = 0; i < newClassNames.length; ++i) {
							classSelection.add(1);
						}
						npairsFrame.menuBar.updateDeselectMenuConds(npairsFrame, classSelection, newClassNames);
						npairsFrame.setClassSelectUpdated(false); 
					}
					catch (IOException e) {
						JOptionPane.showMessageDialog(null, "Could not open class file [" + classFilename + "]",
								"Error", JOptionPane.ERROR_MESSAGE);
					}
				}
			}
		});
		
		cvaClassFilenameField.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				String classFilename = cvaClassFilenameField.getText().toString();
				String[] newClassNames = null;
				if (classFilename.length() > 0) {
					try {		
						newClassNames = NpairsAnalysisMenuBar.getClassNames(classFilename);
						
					}
					catch (IOException e) {
						JOptionPane.showMessageDialog(null, "Could not open class file " + classFilename, "Error",
								JOptionPane.ERROR_MESSAGE);
						return;
					}
				}
				else {  // no class filename entered; use sessionfile conditions as classes
					JOptionPane.showMessageDialog(null, "Using session file conditions as CVA classes.",
							"Message", JOptionPane.INFORMATION_MESSAGE);
					String sessionFileName = "";
					try {
						sessionFileName = npairsFrame.npairsSessionProfilePanel.sessionProfiles.get(0)[0];
						newClassNames = NpairsAnalysisFrame.getConditionNames(sessionFileName);
					}	
					catch (ArrayIndexOutOfBoundsException e) {
						// no session file info loaded yet
					}
					catch (FileNotFoundException e) {
						JOptionPane.showMessageDialog(null, "Could not find NPAIRS sessionfile "
								+ sessionFileName + ".", "Error", JOptionPane.ERROR_MESSAGE);
						return;
					}
					catch (IOException e) {
						JOptionPane.showMessageDialog(null, "Could not load NPAIRS sessionfile " + sessionFileName + 
								".", "Error", JOptionPane.ERROR_MESSAGE);
						return;
					}
				}
				
				// select all classes by default
				Vector<Integer> classSelection = null;
				try {
					classSelection = new Vector<Integer>(newClassNames.length);
					for (int i = 0; i < newClassNames.length; ++i) {
						classSelection.add(1);
					}
				}
				catch (NullPointerException npe) { 
					// no session file info loaded yet
				}
				
				npairsFrame.menuBar.updateDeselectMenuConds(npairsFrame, classSelection, newClassNames);
				npairsFrame.setClassSelectUpdated(false); 
			}
		});

		JPanel modelOptsPane = new JPanel(new GridLayout(1, 1 + modelOptions.length));
		modelOptsPane.add(preprocOptsPane);
		for (int i = 0; i < modelOptions.length; ++i) {
			modelOptsPane.add(modelOptions[i]);
		}

		JPanel loadCVAFileBox = new JPanel(new BorderLayout(5, 0));
		loadCVAFileBox.add(cvaClassFilenameLabel, BorderLayout.WEST);
		loadCVAFileBox.add(cvaClassFilenameField, BorderLayout.CENTER);
		loadCVAFileBox.add(browseButt, BorderLayout.EAST);
		
		mainModelOptsPane.setBorder(BorderFactory.createTitledBorder("NPAIRS Analysis Modelling Options"));
		mainModelOptsPane.add(modelOptsPane, BorderLayout.NORTH);
		mainModelOptsPane.add(normPCChkBox, BorderLayout.CENTER);
		mainModelOptsPane.add(loadCVAFileBox, BorderLayout.SOUTH);
		mainModelOptsPane.setPreferredSize(new Dimension(500, 150));
		
		add(mainModelOptsPane);
	}
	
	public void setAnalysisOptions(boolean doGLM, boolean doPCA, boolean doCVA, String splitObjType) {
		
			modelOptions[0].setSelected(doGLM);
			modelOptions[1].setSelected(doPCA);
			modelOptions[2].setSelected(doCVA);
			browseButt.setEnabled(doCVA && splitObjType.equals("Session (default)"));
			cvaClassFilenameLabel.setEnabled(doCVA && splitObjType.equals("Session (default)"));
			cvaClassFilenameField.setEnabled(doCVA && splitObjType.equals("Session (default)"));
			
	}

	public ModelOptionsPanel setPreprocOptions(boolean doMSR) {
		preprocOptions[0].setSelected(doMSR);
		String splitObjType = (String)npairsFrame.resamplingOptionsPanel.
			splitObjChoice.getSelectedItem();
		if (splitObjType.equals("Run")) {
			preprocOptions[0].setEnabled(false);
		}
		else {
			preprocOptions[0].setEnabled(true);
		}
		return this;
	}	
}


@SuppressWarnings("serial")
final class ResamplingPanel extends JPanel {

	protected JTextField numSplitsField = null;
	protected JTextField splitsFilenameField = null;
	protected JTextField numBootsField = null;
	final JCheckBox resampChkBox;
	final JCheckBox loadSplitsChkBox;
	protected JLabel splitsFileFieldLabel;
	protected JLabel numSplitsLabel;
	final JRadioButton splitHalfXValidButton;
	final JRadioButton bootstrapButton;
	final JButton splitsFileBrowseButt; 
	final JTextField[] splitPartition;
	final JLabel splitPartLabel;
	final NpairsAnalysisFrame npairsFrame;
	final JComboBox splitObjChoice; 
	final JLabel splitObjLabel;
	final String[] splitObjTypes = {"Session (default)", "Run"};

	public ResamplingPanel(NpairsAnalysisFrame frame) {
		this.npairsFrame = frame;
		splitHalfXValidButton = new JRadioButton("Split-half cross-validation");
		bootstrapButton = new JRadioButton("Bootstrap");
		ButtonGroup bGroup = new ButtonGroup();
		bGroup.add(splitHalfXValidButton);
		bGroup.add(bootstrapButton);
		
		// disable technique(s) not yet implemented
		bootstrapButton.setEnabled(false);
		splitHalfXValidButton.setEnabled(true);

		resampChkBox = new JCheckBox("Resample data");
		resampChkBox.setEnabled(true);
		resampChkBox.setSelected(true);
		splitHalfXValidButton.setSelected(true);
		resampChkBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (resampChkBox.isSelected()) {
					splitHalfXValidButton.setEnabled(true);
					splitHalfXValidButton.setSelected(true);
					numSplitsField.setEnabled(true);
					numSplitsLabel.setEnabled(true);
					splitPartLabel.setEnabled(true);
					splitPartition[0].setEnabled(true);
					splitPartition[1].setEnabled(true);			
					// For now, only allow choice of split objects
					// if doing Block NPAIRS analysis. Splitting by
					// run doesn't make sense when loading datamats
					// unless runs aren't merged within each datamat;
					// we're not allowing this possibility right now.
					if (npairsFrame.isBlocked) {
						splitObjLabel.setEnabled(true);
						splitObjChoice.setEnabled(true);
					}
					else {
						splitObjChoice.setEnabled(false);
						splitObjLabel.setEnabled(false);
					}
					loadSplitsChkBox.setEnabled(true);
					if (loadSplitsChkBox.isSelected()) {
						splitsFilenameField.setEnabled(true);
						splitsFileBrowseButt.setEnabled(true);
						splitsFileFieldLabel.setEnabled(true);
						splitPartLabel.setEnabled(false);
						splitPartition[0].setEnabled(false);
						splitPartition[1].setEnabled(false);
					}
				}
				else {
					splitHalfXValidButton.setSelected(false);
					splitHalfXValidButton.setEnabled(false);
					numSplitsField.setEnabled(false);
					numSplitsLabel.setEnabled(false);
					splitsFileFieldLabel.setEnabled(false);
					splitsFilenameField.setEnabled(false);
					splitsFileBrowseButt.setEnabled(false);
					loadSplitsChkBox.setEnabled(false);
					splitPartLabel.setEnabled(false);
					splitPartition[0].setEnabled(false);
					splitPartition[1].setEnabled(false);
					splitObjLabel.setEnabled(false);
					splitObjChoice.setEnabled(false);
				}
			}
		});

		JPanel resampTypePane = new JPanel(new GridLayout(1, 2));
		resampTypePane.setBorder(BorderFactory.createEtchedBorder());
		resampTypePane.add(splitHalfXValidButton);
		resampTypePane.add(bootstrapButton);
		
		splitObjChoice = new JComboBox(splitObjTypes);
		splitObjLabel = new JLabel("Split by: ");
		// disable if analysis is event-related (i.e., if data is loaded from datamats):
		if (!npairsFrame.isBlocked) {
			splitObjLabel.setEnabled(false);
			splitObjChoice.setEnabled(false);
		}
		splitObjChoice.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				npairsFrame.resamplingOptionsPanel.updateSplitPartitions();
				String splitObjType = (String)splitObjChoice.getSelectedItem();
				npairsFrame.resamplingOptionsPanel.updateModelOptions(splitObjType);
			}
		});

		
		numSplitsLabel = new JLabel("No. of splits");
		numSplitsField = new JTextField("500");
		numSplitsField.setEnabled(true);
		numSplitsLabel.setLabelFor(numSplitsField);
		
		splitPartLabel = new JLabel("Split partition");
		splitPartition = new JTextField[2];
		splitPartition[0] = new JTextField("");
		splitPartition[1] = new JTextField("");
		splitPartLabel.setLabelFor(splitPartition[0]);
	
		loadSplitsChkBox = new JCheckBox("Load splits");
		loadSplitsChkBox.setEnabled(true);
		loadSplitsChkBox.setSelected(false);
		loadSplitsChkBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (loadSplitsChkBox.isSelected()) {
					splitsFilenameField.setEnabled(true);
					splitsFileFieldLabel.setEnabled(true);
					splitsFileBrowseButt.setEnabled(true);
					splitPartLabel.setEnabled(false);
					splitPartition[0].setEnabled(false);
					splitPartition[1].setEnabled(false);
				}
				else {
					splitsFilenameField.setEnabled(false);
					splitsFileFieldLabel.setEnabled(false);
					splitsFileBrowseButt.setEnabled(false);
					splitPartLabel.setEnabled(true);
					splitPartition[0].setEnabled(true);
					splitPartition[1].setEnabled(true);
					
				}
			}
		});
		
		splitsFileFieldLabel = new JLabel ("Splits file name");
		splitsFileFieldLabel.setEnabled(false);
		splitsFilenameField = new JTextField(3);
		splitsFilenameField.setEnabled(false);
		splitsFileFieldLabel.setLabelFor(splitsFilenameField);
		
		splitsFileBrowseButt = new JButton("Browse", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
		splitsFileBrowseButt.setIconTextGap(15);
		splitsFileBrowseButt.setEnabled(false);
		
		splitsFileBrowseButt.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				chooser.setFileFilter(new NpairsSplitsInfoFileFilter());
				int option = chooser.showDialog(ResamplingPanel.this, "Load splits info file...");
				if(option == JFileChooser.APPROVE_OPTION) {
					if(!chooser.getSelectedFile().getName().contains(".")) {
						splitsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath() + 
							".vols");
					} 
					else {
						splitsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
					}
				}
			}
		});
		
		// Lay out the top panel.
//		boolean shouldFill = true;
//		boolean shouldWeightX = true;
		GridBagConstraints c = new GridBagConstraints();
//		if (shouldFill) {
			//natural height, maximum width
			c.fill = GridBagConstraints.HORIZONTAL;
//		}
//		if (shouldWeightX) {
			c.weightx = 0.5;
//		}
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 0;
		JPanel topPane = new JPanel(new GridBagLayout());
//		topPane.add(resampChkBox, c);
		c.gridx = 1;
		topPane.add(splitObjLabel);
		c.gridx = 2;
		topPane.add(splitObjChoice, c);
//		c.gridx = 0;
//		c.gridy = 1;
//		topPane.add(resamplePane, c);

		// Lay out the next panel.
		JPanel partPane = new JPanel(new GridLayout(1,2));
		partPane.add(splitPartition[0]);
		partPane.add(splitPartition[1]);
		JPanel middlePane = new JPanel(new GridLayout(1,5));
		middlePane.add(numSplitsLabel);
		middlePane.add(numSplitsField);
		middlePane.add(new JLabel(""));
		middlePane.add(splitPartLabel);
		middlePane.add(partPane);
		
		
		// Lay out the bottom panel.
		JPanel bottomPane = new JPanel (new GridLayout(1,4));
		bottomPane.add(loadSplitsChkBox);
		bottomPane.add(splitsFilenameField);
		bottomPane.add(splitsFileFieldLabel);
		bottomPane.add(splitsFileBrowseButt);
		
		JPanel mainResampPane = new JPanel(new GridLayout(4, 1));
		mainResampPane.add(topPane);
		mainResampPane.add(resampTypePane);
		mainResampPane.add(middlePane);
		mainResampPane.add(bottomPane);
		mainResampPane.setBorder(BorderFactory.createTitledBorder("Resampling Options"));
		mainResampPane.setPreferredSize(new Dimension(500, 140));
		
		add(mainResampPane);
	}
	
	public void setResamplingOptions(boolean doSplitHalfXValid, boolean doBootstrap, 
				int numSplits, String splitsInfoFilename, int[] splitPartitionInfo, 
				String splitObjType) {
		boolean doResampling = false;	
		if (doSplitHalfXValid) {
				doResampling = true;
				splitHalfXValidButton.setSelected(true);
		}
		if (doBootstrap) {
			if (doResampling) {
				JOptionPane.showMessageDialog(null, "Cannot do split-half and bootstrap resampling "
						+ "simultaneously", "Error", JOptionPane.ERROR_MESSAGE);
	        	return;
			}
			doResampling = true;
			bootstrapButton.setSelected(true);
		}
		resampChkBox.setSelected(doResampling);
		if (doResampling) {
			numSplitsField.setEnabled(true);
			numSplitsLabel.setEnabled(true);
			splitObjChoice.setEnabled(true);
			splitObjChoice.setSelectedItem(splitObjType);
			updateModelOptions(splitObjType);
			if (splitPartitionInfo == null) {
				updateSplitPartitions();
			}
		}
		boolean loadSplits = false;
		if (splitsInfoFilename != "") {
			loadSplits = true;
		}
		loadSplitsChkBox.setSelected(loadSplits);
		numSplitsField.setText(Integer.toString(numSplits));
		splitsFilenameField.setText(splitsInfoFilename);
		if (loadSplits) {
			splitsFilenameField.setEnabled(true);
			splitsFileFieldLabel.setEnabled(true);
			splitsFileBrowseButt.setEnabled(true);
			splitPartLabel.setEnabled(false);
			splitPartition[0].setEnabled(false);
			splitPartition[1].setEnabled(false);
		}
		if (splitPartitionInfo != null) {
			splitPartition[0].setText(Integer.toString(splitPartitionInfo[0]));
			splitPartition[1].setText(Integer.toString(splitPartitionInfo[1]));
		}
		
	}
	
	public void updateSplitPartitions() {
		String part1 = "";
		String part2 = "";
		try {
			String splitObjType = (String)splitObjChoice.getSelectedItem();
			int nSplitObj = getNumSplitObj(splitObjType);
			part1 = Integer.toString(nSplitObj / 2);
			part2 = Integer.toString(nSplitObj - (nSplitObj / 2));
		}
		catch (IOException ioe) {
			JOptionPane.showMessageDialog(null, ioe.getMessage(), 
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		splitPartition[0].setText(part1);
		splitPartition[1].setText(part2);
	}
	
	public void updateNVols(Vector<String> sessionProfiles) {
		
	}

	
	protected int getNumSplitObj(String splitObjType) throws IOException {
		int nSplitObj = 0;
		if (splitObjType == "Session (default)") {
			nSplitObj = getNumSessFiles();
		}
		else if (splitObjType == "Run") {
			nSplitObj = getNumRuns();
		}
		return nSplitObj;	 
	}
	
	void updateModelOptions(String splitObjType) {
		if (splitObjType.equals("Session (default)")) {
//			npairsFrame.npairsAnalysisTypeOptionsPanel.
//				preprocOptions[0].setSelected(true);
			npairsFrame.npairsAnalysisTypeOptionsPanel.
				preprocOptions[0].setEnabled(true);
			npairsFrame.npairsAnalysisTypeOptionsPanel.cvaClassFilenameLabel.setEnabled(true);
			npairsFrame.npairsAnalysisTypeOptionsPanel.cvaClassFilenameField.setEnabled(true);
			npairsFrame.npairsAnalysisTypeOptionsPanel.browseButt.setEnabled(true);
			
		}
		else if (splitObjType.equals("Run")) {
			// TODO: fix MSR bug when split object == Run
//			npairsFrame.npairsAnalysisTypeOptionsPanel.
//				preprocOptions[0].setSelected(false); 
			npairsFrame.npairsAnalysisTypeOptionsPanel.
				preprocOptions[0].setEnabled(false); 
			npairsFrame.npairsAnalysisTypeOptionsPanel.cvaClassFilenameLabel.setEnabled(false);
			npairsFrame.npairsAnalysisTypeOptionsPanel.cvaClassFilenameField.setEnabled(false);
			npairsFrame.npairsAnalysisTypeOptionsPanel.browseButt.setEnabled(false);
		}
	}
	
	
	protected int getNumRuns() throws IOException {
		int nRuns = 0;
		int nGrp = npairsFrame.npairsSessionProfilePanel.
			model.getRowCount();
		for (int grp = 0; grp < nGrp; ++grp) {
			String[] sessProfs = npairsFrame.npairsSessionProfilePanel.
				sessionProfiles.get(grp);
			for (int sf = 0; sf < sessProfs.length; ++sf) {
				try {
					SessionProfile currSF = SessionProfile.loadSessionProfile(
							sessProfs[sf], npairsFrame.isBlocked, true);
					nRuns += currSF.getNumInclRuns();
//						System.out.println("No. runs: " + nRuns);
				}
				catch (IOException ioe) {
					throw new IOException("Could not load sessionfile " +
							sessProfs[sf]);
				}
			}
		}
		return nRuns;
	}

	protected int getNumSessFiles() {
		int nSessFiles = 0;
		int nGrp = npairsFrame.npairsSessionProfilePanel.
			model.getRowCount();
		for (int grp = 0; grp < nGrp; ++grp) {
			nSessFiles += npairsFrame.npairsSessionProfilePanel.
				sessionProfiles.get(grp).length;
		}
		return nSessFiles;
	}
}
	

@SuppressWarnings("serial")
final class DataReductionOptionsPanel extends JPanel {

	protected JTextField reductionFactorField = null;	
//	protected JTextField noPCsField = null;
//	protected JLabel nScansLabel = null; 
	final JLabel initFeatSelLabel;
	
	final JRadioButton initEVDRegButton; // regular EVD
	final JRadioButton initEVDNormedButton; // unweighted (all sing vals == 1) EVD
	final ButtonGroup initEVDButtGrp;
	final JCheckBox loadEVDChkBox;
	protected JTextField evdFilePrefField;
	final JButton evdBrowseButt;
//	final JLabel evdFilePrefLabel;
	
	protected JTextField pcsInSplitCVAField = null;
	protected JTextField pcsInFullDataCVAField = null;
	final JLabel numPCsInSplitCVALabel;
	final JLabel numPCsInFullDataCVALabel;
	final JCheckBox setPCRange;
	protected JTextField sField;
	protected JTextField rField;
	final JLabel rLab;
	final JLabel sLab;
	protected JTextField fullDataMultFactField;
	final JLabel fullDataMultFactLab;
	protected JCheckBox setAutoPCRange;
	
	
	
	public DataReductionOptionsPanel() {
		JPanel mainDataRedPane = new JPanel(new BorderLayout(5, 0));

		initFeatSelLabel = new JLabel("Init. feat. selection");
		initEVDRegButton = new JRadioButton("Eigenvalue Decomposition (EVD)");
//		initEVDButton.setToolTipText("Eigenvalue Decomposition");
		initEVDNormedButton = new JRadioButton("Normed EVD");
		initEVDButtGrp = new ButtonGroup();
		initEVDButtGrp.add(initEVDRegButton);
		initEVDButtGrp.add(initEVDNormedButton);
		initEVDNormedButton.setToolTipText("All singular values are set to 1 in normed EVD");
		loadEVDChkBox = new JCheckBox("Load EVD");
//		evdFilePrefLabel = new JLabel("EVD file prefix");
		evdBrowseButt = new JButton("Browse", new ImageIcon(this.getClass().
				getResource("/toolbarButtonGraphics/general/Import16.gif")));
		evdFilePrefField = new JTextField();
		
		// default is to do initial SVD
//		initFeatSelChkBox.setSelected(true);
//		initFeatSelLabel.setEnabled(true);
		initEVDRegButton.setSelected(true);
//		initEVDButton.setEnabled(true);		
//		loadEVDChkBox.setSelected(false);
//		evdFilePrefField.setEnabled(false);
//		evdBrowseButt.setEnabled(false);
//		evdFilePrefLabel.setEnabled(false);
		
//		initFeatSelChkBox.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent e) {
//				if (initFeatSelChkBox.isSelected()) {
//					initSVDButton.setEnabled(true);
//					initSVDButton.setSelected(true);
//					reductionFactorField.setEnabled(true);
//					loadSVDChkBox.setEnabled(true);
//				}
//				else {
//					initSVDButton.setSelected(false);
//					initSVDButton.setEnabled(false);
//					reductionFactorField.setEnabled(false);
//					loadSVDChkBox.setSelected(false);
//					loadSVDChkBox.setEnabled(false);
//					if (!loadSVDChkBox.isSelected()) {
//						svdFilePrefField.setEnabled(false);
//						svdBrowseButt.setEnabled(false);
//						svdFilePrefLabel.setEnabled(false);
//					}
//				}
//			}
//		});
		
//		initEVDButton.addActionListener(new ActionListener() {
//			public void actionPerformed(ActionEvent e) {
//				if (initEVDButton.isSelected()) {
//					loadEVDChkBox.setEnabled(true);
//				}
//				else {
//					loadEVDChkBox.setSelected(false);
//					loadEVDChkBox.setEnabled(false);
//					evdFilePrefField.setEnabled(false);
//					evdBrowseButt.setEnabled(false);
//					evdFilePrefLabel.setEnabled(false);
//				}
//			}
//			
//		});
		
		loadEVDChkBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (loadEVDChkBox.isSelected()) {
					evdFilePrefField.setEnabled(true);
					evdBrowseButt.setEnabled(true);
//					evdFilePrefLabel.setEnabled(true);
				}
				else {
					evdFilePrefField.setEnabled(false);
					evdBrowseButt.setEnabled(false);
//					evdFilePrefLabel.setEnabled(false);
				}
			}
		});
		
		evdBrowseButt.setIconTextGap(15);
		evdBrowseButt.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				chooser.setFileFilter(new NpairsEVDFileFilter());
				int option = chooser.showDialog(DataReductionOptionsPanel.this, "Load EVD files");
				if(option == JFileChooser.APPROVE_OPTION) {
					File f = chooser.getSelectedFile();
					if (!f.getName().contains(".EVD.")) {
						evdFilePrefField.setText(f.getAbsolutePath());
					} 
					else {
						// strip suffix, including .EVD.
						int i = f.getAbsolutePath().lastIndexOf(".EVD.");
						evdFilePrefField.setText(f.getAbsolutePath().substring(0, i));
					}
				}
			}
		});
		
		setAutoPCRange = new JCheckBox("Auto PC range");
		setAutoPCRange.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				if (setAutoPCRange.isSelected()) {
					pcsInSplitCVAField.setEnabled(false);
					pcsInFullDataCVAField.setEnabled(false);
					numPCsInSplitCVALabel.setEnabled(false);
					numPCsInFullDataCVALabel.setEnabled(false);

					rLab.setEnabled(false);
					rField.setEnabled(false);
					rField.setText("");
					sLab.setEnabled(false);
					sField.setEnabled(false);
					sField.setText(((Integer)NpairsjSetupParams.AUTO_PC_STEP_SIZE).toString());
					fullDataMultFactLab.setEnabled(false);
					fullDataMultFactField.setEnabled(false);
					fullDataMultFactField.setText("1.0");
					
				}
				else {	
					rLab.setEnabled(true);
					rField.setEnabled(true);
					sLab.setEnabled(true);
					sField.setEnabled(true);
					fullDataMultFactLab.setEnabled(true);
					fullDataMultFactField.setEnabled(true);
				}
			}
		});
		setPCRange = new JCheckBox("Set PC range");
		setPCRange.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				if (setPCRange.isSelected()) {
					setAutoPCRange.setEnabled(true);
					setAutoPCRange.setSelected(false);
					pcsInSplitCVAField.setEnabled(false);
					pcsInFullDataCVAField.setEnabled(false);
					numPCsInSplitCVALabel.setEnabled(false);
					numPCsInFullDataCVALabel.setEnabled(false);
					
					rLab.setEnabled(true);
					rField.setEnabled(true);				
					sLab.setEnabled(true);
					sField.setEnabled(true);
					fullDataMultFactLab.setEnabled(true);
					fullDataMultFactField.setEnabled(true);
					fullDataMultFactField.setText("1.0"); // default value 
				}
				else {
					setAutoPCRange.setEnabled(false);
					setAutoPCRange.setSelected(false);
					pcsInSplitCVAField.setEnabled(false);
					pcsInSplitCVAField.setEnabled(true);
					pcsInFullDataCVAField.setEnabled(true);
					numPCsInSplitCVALabel.setEnabled(true);
					numPCsInFullDataCVALabel.setEnabled(true);
					
					rLab.setEnabled(false);
					rField.setEnabled(false);
					sLab.setEnabled(false);
					sField.setEnabled(false);
					fullDataMultFactLab.setEnabled(false);
					fullDataMultFactField.setEnabled(false);
				}
			}
		});

		JLabel reductionFactorLabel = new JLabel("Data Reduction Factor");
//		JLabel EVDNoPCsLabel = new JLabel("No. dims (PCs) to keep in EVD");
//		nScansLabel = new JLabel("No. input scans: 0");
		reductionFactorField = new JTextField();
//		noPCsField = new JTextField();
//	//	reductionFactorField.setColumns(3);
		reductionFactorField.setText("0.3");
		
//		// TODO: Move num vols panel into session file panel and clean up
//		// how it is laid out. Added to data reduction panel only because 
//		// it was easiest place to figure out how to fit it into GUI layout
//		final JLabel numVolsLab = new JLabel("No. scans:");
//		final JTextField numVolsField = new JTextField();
//		numVolsField.setEnabled(false);
//		JPanel numVolsPanel = new JPanel();
//		numVolsPanel.setLayout(new GridLayout(0,1));
//		numVolsPanel.setBorder(BorderFactory.createEtchedBorder());
//		numVolsPanel.add(numVolsLab);
//		numVolsPanel.add(numVolsField);
//		JPanel mainNumVolsPanel = new JPanel();
//		mainNumVolsPanel.setLayout(new BorderLayout());
//		mainNumVolsPanel.add(numVolsPanel, BorderLayout.CENTER);
//		mainNumVolsPanel.add(new JPanel(),BorderLayout.NORTH);
		
		numPCsInSplitCVALabel = new JLabel("Set of PCs to use in split CVA ");
		pcsInSplitCVAField = new JTextField();
		pcsInSplitCVAField.setText("0");
		pcsInSplitCVAField.setEnabled(true);

		numPCsInFullDataCVALabel = new JLabel("Set of PCs to use in full-data CVA");
		pcsInFullDataCVAField = new JTextField();
		pcsInFullDataCVAField.setText("0");

	   
		rLab = new JLabel("Range");
		rField = new JTextField(6);
		sLab = new JLabel("Step");
		sField = new JTextField(4);
		
//		final boolean shouldFill = true;
//	    final boolean shouldWeightX = true;
	    GridBagConstraints c = new GridBagConstraints();
//		if (shouldFill) {
		//natural height, maximum width
		c.fill = GridBagConstraints.HORIZONTAL;
//		}
//		if (shouldWeightX) {
		c.weightx = 0.5;
//		}
		c.fill = GridBagConstraints.HORIZONTAL;
        c.gridx = 0;
        c.gridy = 0;
        
        JPanel splitPCRangePane = new JPanel(new GridBagLayout());
        splitPCRangePane.add(setPCRange,c);
        c.gridx = 1;
        splitPCRangePane.add(setAutoPCRange,c);
        c.gridx = 2;
		splitPCRangePane.add(rField, c);
		c.gridx = 3;
		splitPCRangePane.add(rLab, c);
//		JPanel sPane = new JPanel(new GridLayout(1,3,0,0));
		c.gridx = 4;
		splitPCRangePane.add(sField,c);
		c.gridx = 5;
		splitPCRangePane.add(sLab,c);
//		sPane.add(sField);
//		sPane.add(sLab);
		
		
		fullDataMultFactLab = new JLabel("Proportion of PC range to use in full-data CVA");
		fullDataMultFactField = new JTextField(5);
		
		// Lay out pc range fields in a panel
//		JPanel splitPCRangePane = new JPanel(new FlowLayout());
//		splitPCRangePane.add(setPCRange);
//		splitPCRangePane.add(rPane);
//		splitPCRangePane.add(sPane);
//		splitPCRangePane.setBorder(BorderFactory.createEtchedBorder());
		splitPCRangePane.setBorder(BorderFactory.createEtchedBorder());
		JPanel fullDataPCRangePane = new JPanel(new FlowLayout());
		fullDataPCRangePane.add(fullDataMultFactField);
		fullDataPCRangePane.add(fullDataMultFactLab);
		JPanel pcRangePane = new JPanel(new GridLayout(2,1));
//		pcRangePane.add(splitPCRangePane);
		pcRangePane.add(splitPCRangePane);
		pcRangePane.add(fullDataPCRangePane);
		
		rField.setEnabled(false);
		rLab.setEnabled(false);
		sField.setEnabled(false);
		sLab.setEnabled(false);
		fullDataMultFactField.setEnabled(false);
		fullDataMultFactField.setEnabled(false);
		setAutoPCRange.setEnabled(false);
		
		
		// Lay out EVD load fields in 2 panels
		JPanel loadSVDLeftPanel = new JPanel(new GridLayout(1,2));
		JPanel loadSVDRightPanel = new JPanel(new GridLayout(1,2));
		loadSVDLeftPanel.add(loadEVDChkBox);
//		loadSVDLeftPanel.add(evdFilePrefLabel);
		loadSVDRightPanel.add(evdFilePrefField);
		loadSVDRightPanel.add(evdBrowseButt);
		
		// Lay out the labels in a panel.
		JPanel rightPane = new JPanel(new GridLayout(0, 1, 0, 4));
		JPanel initFeatPane = new JPanel(new GridLayout(0, 2, 0, 5));
		initFeatPane.setBorder(BorderFactory.createEtchedBorder());
		initFeatPane.add(initEVDRegButton);
		initFeatPane.add(initEVDNormedButton);
//		rightPane.add(initFeatPane);
//		rightPane.add(new JLabel(""));
		rightPane.add(loadSVDRightPanel);
		rightPane.add(reductionFactorLabel);
//		rightPane.add(EVDNoPCsLabel);
		rightPane.add(numPCsInSplitCVALabel);
		rightPane.add(numPCsInFullDataCVALabel);
		
		// Lay out the text fields in a panel.
		JPanel leftPane = new JPanel(new GridLayout(0, 1, 0, 4));
//		leftPane.add(initFeatSelLabel);
		leftPane.add(loadSVDLeftPanel);
		leftPane.add(reductionFactorField);
//		leftPane.add(noPCsField);
		leftPane.add(pcsInSplitCVAField);
		leftPane.add(pcsInFullDataCVAField);	

		mainDataRedPane.add(initFeatPane, BorderLayout.NORTH);
		mainDataRedPane.add(leftPane, BorderLayout.WEST);
//		mainDataRedPane.add(nScansLabel, BorderLayout.EAST);
		mainDataRedPane.add(rightPane, BorderLayout.CENTER);
//		mainDataRedPane.add(mainNumVolsPanel, BorderLayout.EAST);
		mainDataRedPane.add(pcRangePane, BorderLayout.SOUTH);
		mainDataRedPane.setBorder(BorderFactory.createTitledBorder("Data Reduction Options"));
		mainDataRedPane.setPreferredSize(new Dimension(500, 240));

		add(mainDataRedPane);
	}
	
	

	 /** 
	 * @param doRegularEVD if false, do unweighted EVD instead
	 * @param loadEVD
	 * @param evdFilePrefix
	 * @param dataRedFactor
	 * @param splitPCs
	 * @param fullDataPCs
	 * @param doPCRange
	 * @param pcRange
	 * @param step
	 * @param pcMultFact
	 * @param autoPCRange TODO
	 * @param autoPCRange
	 */
	public void setDataReductionOptions(boolean doRegularEVD, boolean loadEVD, String evdFilePrefix, 
			double dataRedFactor, String splitPCs, String fullDataPCs,
			boolean doPCRange, String pcRange, int step, double pcMultFact, boolean autoPCRange) {
		
		initEVDRegButton.setSelected(doRegularEVD);
		initEVDNormedButton.setSelected(!doRegularEVD);
		loadEVDChkBox.setSelected(loadEVD);
		evdFilePrefField.setEnabled(loadEVD);
		evdFilePrefField.setText(evdFilePrefix);
		evdBrowseButt.setEnabled(loadEVD);
		reductionFactorField.setText(Double.toString(dataRedFactor));
		pcsInSplitCVAField.setEnabled(!doPCRange);
		pcsInSplitCVAField.setText(splitPCs);
		pcsInFullDataCVAField.setEnabled(!doPCRange);
		pcsInFullDataCVAField.setText(fullDataPCs);
		setPCRange.setSelected(doPCRange);
		setAutoPCRange.setEnabled(doPCRange);
		setAutoPCRange.setSelected(autoPCRange);
		sField.setText(new Integer(step).toString());
		fullDataMultFactField.setText(new Double(pcMultFact).toString());
		if (!autoPCRange) {
			rLab.setEnabled(doPCRange);
			rField.setEnabled(doPCRange);
			rField.setText(pcRange);
			sLab.setEnabled(doPCRange);
			sField.setEnabled(doPCRange);
			fullDataMultFactLab.setEnabled(doPCRange);
			fullDataMultFactField.setEnabled(doPCRange);			
		}
		else {
			rLab.setEnabled(false);
			rField.setEnabled(false);
			rField.setText("");
			sLab.setEnabled(false);
			sField.setEnabled(false);
			fullDataMultFactLab.setEnabled(false);
			fullDataMultFactField.setEnabled(false);
		}
			
	}
}




