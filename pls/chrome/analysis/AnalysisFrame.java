package pls.chrome.analysis;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.border.Border;
import javax.swing.table.DefaultTableModel;

import pls.analysis.Analysis;
import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.ContrastsFrame;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.MLFuncs;
import pls.shared.PlsAnalysisSetupFileFilter;
import pls.shared.StreamedProgressDialog;
import pls.shared.StreamedProgressHelper;
import pls.shared.fMRIContrastFileFilter;
import pls.shared.fMRIResultFileFilter;
import pls.shared.fMRISessionFileFilter;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

@SuppressWarnings("serial")
public class AnalysisFrame extends JFrame {
	
	protected final SessionProfilesPanel sessionProfilePanel;
	protected final PermBootPanel permBootPanel;
	protected final PlsOptionPanel plsOptionPanel;
	protected final JTextField resultsFilenameField;
	protected final String extension;
	final AnalysisMenuBar menu;
	
	public AnalysisFrame(final boolean isBlockedFmri) {
		super("PLS Analysis");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	    final JFrame frame = this;
        
        sessionProfilePanel = new SessionProfilesPanel(frame, isBlockedFmri);
        permBootPanel = new PermBootPanel();
        plsOptionPanel = new PlsOptionPanel(this);
        //final NpairsOptionPanel npairsOptionPanel = new NpairsOptionPanel();
        
        menu = new AnalysisMenuBar(frame, 
	    		sessionProfilePanel.conditionSelection, 
	    		sessionProfilePanel.behaviorBlockConditionSelection, 
	    		sessionProfilePanel.sessionProfiles);
	    setJMenuBar(menu);
        
        JPanel containerPane = new JPanel();
        containerPane.setLayout(new BoxLayout(containerPane, BoxLayout.Y_AXIS));
        containerPane.add(sessionProfilePanel);
        containerPane.add(plsOptionPanel);
        containerPane.add(permBootPanel);
        //containerPane.add(npairsOptionPanel);
	    
	    resultsFilenameField = new JTextField();
	    resultsFilenameField.setColumns(12);
		JButton resultsButton = new JButton("Results File Prefix", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
		resultsButton.setIconTextGap(15);
	    
		if (isBlockedFmri) {
			extension = BfMRIResultFileFilter.EXTENSION;
		}
		else {
			extension = fMRIResultFileFilter.EXTENSION;
		}
		
	    resultsButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				
				if(isBlockedFmri){
					chooser.setFileFilter(new BfMRIResultFileFilter());
				}else{
					chooser.setFileFilter(new fMRIResultFileFilter());
				}
				
				int option = chooser.showDialog(AnalysisFrame.this, "Save results to");
				if(option == JFileChooser.APPROVE_OPTION) {
					if(!chooser.getSelectedFile().getName().contains(".")) {
						resultsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath() + extension);
					} else {
						resultsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
					}
				}
			}
		});

	    
	    JPanel resultsPane = new JPanel(new GridLayout(1, 0, 10, 0));
	    resultsPane.add(resultsFilenameField);
	    resultsPane.add(resultsButton);
//	    resultsPane.add(npairsSetupParamFilenameField);
//	    resultsPane.add(npairsSetupParamFileButton);
	    resultsPane.setBorder(BorderFactory.createEmptyBorder(5,0,5,0));
        
		final JButton runButton = new JButton("Run", 
				new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Play16.gif")));
		runButton.setIconTextGap(15);
		runButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				new RunPls(frame, 0, sessionProfilePanel, 
						   permBootPanel, plsOptionPanel, 
						   resultsFilenameField,extension);
			}
		});
		
        JPanel mainPane = new JPanel(new BorderLayout());
		mainPane.add(containerPane, BorderLayout.NORTH);
		mainPane.add(resultsPane, BorderLayout.CENTER);
		mainPane.add(runButton, BorderLayout.SOUTH);
		mainPane.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
		JScrollPane mainScrollPane = new JScrollPane(mainPane);
		add(mainScrollPane);
		
		// Display the window
        pack();
        
        setVisible(true);
		
		//adjusts frame's dimensions
		int verMagicNum = 3; 
		int horMagicNum = 50;
		setSize(getWidth() + horMagicNum, getHeight() - verMagicNum);
		
		// Position the frame on the screen
		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		int x = (int)(screen.getWidth() - getWidth()) / 2;
		int y = (int)(screen.getHeight() - getHeight()) / 2;
		setLocation(x, y);
		
		enableContrastMenu(false);
	 }
	
	void enableContrastMenu(boolean val){
		menu.enableContrastMenu(val);
	}
}

final class RunPls {
	
	public RunPls(JFrame parent, 
				  int imagingType, 
				  SessionProfilesPanel sessionProfilePanel, 
				  PermBootPanel permBootPanel, 
				  PlsOptionPanel plsOptionPanel, 
				  JTextField resultsFilenameField,
				  String extension) {
		
		// Gather input from gui and validate
		Vector<String[]> sessionProfiles = sessionProfilePanel.sessionProfiles;

		if(plsOptionPanel.radioButtons[1].isSelected() || plsOptionPanel.radioButtons[3].isSelected() || plsOptionPanel.radioButtons[4].isSelected()) {
			int count = 0;
			for(String[] sp : sessionProfiles) {
					count += sp.length;
			}
			if(count < 3) {
				JOptionPane.showMessageDialog(null, "Behavior and Multiblock analyses require at least 3 subjects.\nAlso, their behavior data values cannot all be the same for any group.", "Error", JOptionPane.ERROR_MESSAGE);
				return;
			}
		}
		
		int[] conditionSelection = new int[sessionProfilePanel.conditionSelection.size()];
		for(int i = 0; i < conditionSelection.length; i++) {
			conditionSelection[i] = sessionProfilePanel.conditionSelection.get(i).intValue();
		}
		
		int[] behaviorBlockConditionSelection = new int[sessionProfilePanel.behaviorBlockConditionSelection.size()];
		for(int i = 0; i < behaviorBlockConditionSelection.length; i++) {
			behaviorBlockConditionSelection[i] = sessionProfilePanel.behaviorBlockConditionSelection.get(i).intValue();
		}
		
		String behaviorFilename = plsOptionPanel.behaviorFilenameField.getText().trim();
		String contrastFilename = plsOptionPanel.contrastFilenameField.getText().trim();
		
		String resultsFilename = resultsFilenameField.getText().trim();
		
		if(resultsFilename.equals("")) {
			JOptionPane.showMessageDialog(null, "Please enter a results filename.", "Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		int numPermutations = Integer.parseInt(permBootPanel.numPermsField.getText());
		int numBootstraps = Integer.parseInt(permBootPanel.numBootsField.getText());
		double confidenceLevel = Double.parseDouble(permBootPanel.confidenceField.getText());

		/********************/
		if(imagingType != 2){ // this check is done for only fMRI
			
			for(int i = 0; i < sessionProfiles.size(); i++) {
				if(numBootstraps > 0 && sessionProfiles.get(i).length < 3) {
					JOptionPane.showMessageDialog(null, imagingType+" Bootstrap requires 3 or more subjects per group.", "Too few subjects in a group", JOptionPane.ERROR_MESSAGE);
					return;
				}
			}
		}
		/************************/
		int plsType = 0;
		for(int i = 0; i < plsOptionPanel.radioButtons.length; i++) {
			if(plsOptionPanel.radioButtons[i].isSelected()) {
				plsType = i;
			}
		}
		
		// Gather NPAIRS data into one variable
//		NpairsSetupParameters npairsSetup = new NpairsSetupParameters();
//		
//		npairsSetup.reductionfactor = Float.parseFloat(npairsOptionPanel.reductionFactorField.getText());
//		npairsSetup.cvapcset = Integer.parseInt(npairsOptionPanel.numPCsInCVAField.getText());
//		npairsSetup.cvaallpcset = Integer.parseInt(npairsOptionPanel.numPCsInDataField.getText());
//		
		// Start new analysis
//		Analysis worker = new Analysis(imagingType, plsType, sessionProfiles, contrastFilename, behaviorFilename, posthocFilename, resultsFilename, conditionSelection, behaviorBlockConditionSelection, numPermutations, numBootstraps, confidenceLevel, npairsSetup);
		
		try {
			StreamedProgressDialog dialog = new StreamedProgressDialog(parent, numPermutations + numBootstraps + 3);
			StreamedProgressHelper helper = new StreamedProgressHelper();
			PipedOutputStream pos = new PipedOutputStream();
			
			dialog.connectWriter(pos);
			helper.addStream(pos);
			
			//creates the result directory that the result file will be placed into.
			resultsFilename = createResultsDir(resultsFilename,helper,extension);
			if(resultsFilename == null){
				dialog.complete();
				return;
			}
						
			Analysis worker = new Analysis(imagingType, plsType, sessionProfiles, contrastFilename, 
					behaviorFilename, resultsFilename, conditionSelection, 
					behaviorBlockConditionSelection, numPermutations, numBootstraps, confidenceLevel);
			
			dialog.worker = worker;
			
			worker.progress = helper;
			worker.start();
		} catch (IOException ioex) {
			ioex.printStackTrace();
		}
//		new ProgressDialog(parent, numPermutations + numBootstraps + 3, worker);
	}
	
	/**
	 * Create the results file directory that the generated result file will 
	 * be placed into.
	 * @param resultsFilename The text in the 'result file name' text field in
	 * the pls analysis window.
	 * @param output The progress frame where status messages are sent.
	 * @param extension The extension to be appended to the saved result file.
	 * @return
	 */
	String createResultsDir(String resultsFilename, 
			StreamedProgressHelper output, String extension){
		
		File path = new File(resultsFilename);
		File parent = path.getParentFile();
		String targetfile = path.getName();
		
		if(!targetfile.endsWith(extension)){
			targetfile += extension;
		}
		
		if(parent == null){
			parent = new File(System.getProperty("user.dir"));
		}
		
		if(!parent.exists()){
			output.postMessage(("Creating result dir: " + parent.getAbsolutePath()));
			if(!parent.mkdir()){
				output.printError("Unable to create results dir.");
				return null;
			}
		}
		if(!parent.canWrite()){
			output.printError("Cannot write to result dir.");
			return null;
		}
		output.postMessage("\nSaving result file: " + targetfile);
		return parent.getAbsolutePath() + File.separator + targetfile;
	}
}

final class SessionProfilesPanel extends JPanel {
	
	protected Vector<String[]> sessionProfiles = new Vector<String[]>();
	
	protected Vector<Integer> conditionSelection = new Vector<Integer>();
	
	protected Vector<Integer> behaviorBlockConditionSelection = new Vector<Integer>();
	
	protected final DefaultTableModel model;
	
	protected final SelectSessionFilesDialog selectDialog;
	
	public SessionProfilesPanel(JFrame parent, final boolean isBlockedFmri) {
		
		JPanel mainPanel = new JPanel(new BorderLayout());
		
		model = new DefaultTableModel();
		final JTable jt = new JTable(model){
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
	    
	    if (isBlockedFmri) {
			selectDialog = new SelectSessionFilesDialog(parent, new BfMRISessionFileFilter(), sessionProfiles, jt, model);
		} else {
			selectDialog = new SelectSessionFilesDialog(parent, new fMRISessionFileFilter(), sessionProfiles, jt, model);
		}
        selectDialog.setSize(640, 480);
        selectDialog.setResizable(false);
		
		JPanel buttonPanel = new JPanel();
		final JButton addButton = new JButton("Add Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		buttonPanel.add(addButton);
		final JButton editButton = new JButton("Edit Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Edit16.gif")));
		buttonPanel.add(editButton);
		editButton.setEnabled(false);
		final JButton deleteButton = new JButton("Delete Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Delete16.gif")));
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
					conditionSelection.removeAllElements();
					behaviorBlockConditionSelection.removeAllElements();
				}
			}
		});
		
	    Border border = BorderFactory.createTitledBorder("Session Profiles");
	    mainPanel.setBorder(border);
		mainPanel.setPreferredSize(new Dimension(600, 150));
		add(mainPanel);
	}
	
	/**
	 * Update the session file profiles table with the newly loaded session 
	 * files.
	 * @param sessionProfiles
	 */
	public void setSessProfiles(Vector<String[]> sessionProfiles) {
		String fileList;
		
		model.setRowCount(0);
		for(String[] group : sessionProfiles){
			fileList = "";
			
			for(String sf : group){
				fileList += sf + "; ";
			}
			model.addRow(new Object[]{model.getRowCount() + 1, fileList});
		}
	}
	
	public Vector<String[]> getSessProfiles(){
		return sessionProfiles;
	}
	
	public void setConditionSelection(Vector<Integer> cs) {
		conditionSelection.removeAllElements();
		for(Integer i : cs) conditionSelection.add(i);
		
	}
	
	public Vector<Integer> getConditionSelection(){
		return conditionSelection;
	}
	
	public void setBehavBlockConditionSelection(Vector<Integer> behaviorBlockConditionSelection) {
		this.behaviorBlockConditionSelection = behaviorBlockConditionSelection;
	}
}

final class PlsOptionPanel extends JPanel {
	
	JRadioButton[] radioButtons = null;
	
	JTextField contrastFilenameField  = new JTextField();
	
	JTextField behaviorFilenameField  = new JTextField();
	
	public PlsOptionPanel(final AnalysisFrame baseFrame) {
		JPanel mainPanel = new JPanel(new BorderLayout());
	    ButtonGroup group = new ButtonGroup();
	    radioButtons = new JRadioButton[5];
	
	    radioButtons[0] = new JRadioButton("Mean-Centering PLS");
	    radioButtons[0].setSelected(true);
	    radioButtons[1] = new JRadioButton("Behavior PLS");
	    radioButtons[2] = new JRadioButton("Non-Rotated Task PLS"); 
	    radioButtons[3] = new JRadioButton("Multiblock PLS");
	    radioButtons[3].setEnabled(false); //TODO: re-enable when it works
	    radioButtons[4] = new JRadioButton("Non-Rotated Behavior PLS");
//	    radioButtons[5] = new JRadioButton("NPAIRS with CVA");
	    
	    for(int i = 0; i < radioButtons.length; i++) {
	    	group.add(radioButtons[i]);
	    }
	    
		JPanel radioBox = new JPanel(new GridLayout(2, 2));
	    Border border = BorderFactory.createEtchedBorder();
	    radioBox.setBorder(border);

	    radioBox.add(radioButtons[0]);
	    radioBox.add(radioButtons[1]);
	    radioBox.add(radioButtons[4]);
	    radioBox.add(radioButtons[2]);
	    radioBox.add(radioButtons[3]);
	    
	    contrastFilenameField.setColumns(12);
	    contrastFilenameField.setEnabled(true);
		final JButton loadContrastButton = new JButton("Load Contrast Data", 
				new ImageIcon(
						this.getClass().getResource(
								"/toolbarButtonGraphics/general/Open16.gif")));
		
		loadContrastButton.setIconTextGap(15);
		loadContrastButton.setEnabled(true);
	    
	    loadContrastButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new fMRIContrastFileFilter());
				
			    int option = chooser.showDialog(PlsOptionPanel.this, 
						"Select contrast data file");
				
			    if (option == JFileChooser.APPROVE_OPTION) {
					contrastFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	    
	    behaviorFilenameField.setColumns(12);
	    behaviorFilenameField.setEnabled(false);
		final JButton behaviorButton = new JButton("Load Behavior Data", 
				new ImageIcon(
						this.getClass().getResource(
								"/toolbarButtonGraphics/general/Open16.gif")));
		
		behaviorButton.setIconTextGap(15);
		behaviorButton.setEnabled(false);
	    
		behaviorButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				int option = chooser.showDialog(PlsOptionPanel.this, "Select behavior data file");
				if (option == JFileChooser.APPROVE_OPTION) {
					behaviorFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
				
		ItemListener activateBehaviour = new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				boolean opposite = (radioButtons[1].isSelected() || radioButtons[3].isSelected() || radioButtons[4].isSelected());
				behaviorButton.setEnabled(opposite);
				behaviorFilenameField.setEnabled(opposite);
				
				if(radioButtons[2].isSelected() || radioButtons[4].isSelected()){
					baseFrame.enableContrastMenu(true);
				}else{
					baseFrame.enableContrastMenu(false);
				}
			}
		};
		
		radioButtons[0].addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent ae){
				baseFrame.enableContrastMenu(false);
			}
		});
		radioButtons[1].addItemListener(activateBehaviour);
		radioButtons[2].addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				boolean opposite = (radioButtons[2].isSelected());
				loadContrastButton.setEnabled(opposite);
				contrastFilenameField.setEnabled(opposite);
				baseFrame.enableContrastMenu(true);
			}
		});
		radioButtons[4].addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				boolean opposite = (radioButtons[4].isSelected());
				loadContrastButton.setEnabled(opposite);
				contrastFilenameField.setEnabled(opposite);
			}
		});
		
		radioButtons[3].addItemListener(activateBehaviour);
		radioButtons[4].addItemListener(activateBehaviour);
		
		
	    JPanel fieldPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    fieldPane.setBorder(BorderFactory.createEmptyBorder(0,10,0,10));
	    fieldPane.add(contrastFilenameField);
	    fieldPane.add(behaviorFilenameField);
	    
	    JPanel buttonPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    buttonPane.setBorder(BorderFactory.createEmptyBorder(0,10,0,10));
	    buttonPane.add(loadContrastButton);
	    buttonPane.add(behaviorButton);
	    
	    JPanel topPane = new JPanel(new GridLayout(0,2));
        topPane.add(fieldPane, BorderLayout.CENTER);
        topPane.add(buttonPane, BorderLayout.LINE_END);
        topPane.setBorder(BorderFactory.createEmptyBorder(0,0,10,0));
		
	    Border border2 = BorderFactory.createTitledBorder("PLS Option");
	    mainPanel.setBorder(border2);
		mainPanel.add(radioBox, BorderLayout.NORTH);
		mainPanel.add(topPane, BorderLayout.SOUTH);
		mainPanel.setPreferredSize(new Dimension(600, 190));
		add(mainPanel);
	}
}


final class PermBootPanel extends JPanel {
	
	protected JTextField numPermsField = null;
	protected JTextField numBootsField = null;
	protected JTextField confidenceField = null;
	
	public PermBootPanel() {
		JPanel mainPane = new JPanel(new GridLayout(0, 1));

	    JLabel numPermsLabel = new JLabel("Number of Permutations");
	    numPermsField = new JTextField();
	    numPermsField.setColumns(5);
	    numPermsField.setText("0");
	    numPermsLabel.setLabelFor(numPermsField);

	    JLabel numBootsLabel = new JLabel("<html>Number of Bootstraps <span style=\"color: red;\">" +
	    								"(Requires at least 3 subjects per group)</html>");
	    numBootsField = new JTextField();
	    numBootsField.setColumns(5);
	    numBootsField.setText("0");
	    numBootsLabel.setLabelFor(numBootsLabel);

	    JLabel confidenceLabel = new JLabel("Confidence Level");
	    confidenceField = new JTextField();
	    confidenceField.setColumns(5);
	    confidenceField.setText("95");
	    confidenceLabel.setLabelFor(confidenceField);
	    
	    // Lay out the labels in a panel.
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 0, 2));
        labelPane.add(numPermsLabel);
        labelPane.add(numBootsLabel);
        labelPane.add(confidenceLabel);
        
        // Layout the text fields in a panel.
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 0, 2));
        fieldPane.add(numPermsField);
        fieldPane.add(numBootsField);
        fieldPane.add(confidenceField);
	    
	    JPanel topPane = new JPanel(new BorderLayout(5, 0));
        topPane.add(fieldPane, BorderLayout.WEST);
        topPane.add(labelPane, BorderLayout.CENTER);
		
	    Border border = BorderFactory.createTitledBorder("Permutations / Bootstraps");
	    mainPane.setBorder(border);
		mainPane.setPreferredSize(new Dimension(600, 90));
	    
	    mainPane.add(topPane);
	    
		add(mainPane);
	}
}


final class NpairsOptionPanel extends JPanel {
	
	protected JTextField reductionFactorField = null;
	protected JTextField numPCsInCVAField = null;
	protected JTextField numPCsInDataField = null;
	
	public NpairsOptionPanel() {
		JPanel mainPane = new JPanel(new GridLayout(0, 1));

	    JLabel reductionFactorLabel = new JLabel("Data Reduction Factor");
	    reductionFactorField = new JTextField();
	    reductionFactorField.setColumns(5);
	    reductionFactorField.setText("1.0");
	    reductionFactorLabel.setLabelFor(reductionFactorField);

	    JLabel numPCsInCVALabel = new JLabel("Set of principal components to use in CVA");
	    numPCsInCVAField = new JTextField();
	    numPCsInCVAField.setColumns(5);
	    numPCsInCVAField.setText("0");
	    numPCsInCVALabel.setLabelFor(numPCsInCVALabel);

	    JLabel numPCsInDataLabel = new JLabel("Set of principal components to use in CVA of all data");
	    numPCsInDataField = new JTextField();
	    numPCsInDataField.setColumns(5);
	    numPCsInDataField.setText("0");
	    numPCsInDataLabel.setLabelFor(numPCsInDataField);
	    
	    // Lay out the labels in a panel.
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 0, 5));
        labelPane.add(reductionFactorLabel);
        labelPane.add(numPCsInCVALabel);
        labelPane.add(numPCsInDataLabel);
        
        // Layout the text fields in a panel.
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 0, 5));
        fieldPane.add(reductionFactorField);
        fieldPane.add(numPCsInCVAField);
        fieldPane.add(numPCsInDataField);
	    
	    JPanel topPane = new JPanel(new BorderLayout(5, 0));
        topPane.add(fieldPane, BorderLayout.WEST);
        topPane.add(labelPane, BorderLayout.CENTER);
		
	    Border border = BorderFactory.createTitledBorder("NPAIRS options");
	    mainPane.setBorder(border);
		mainPane.setPreferredSize(new Dimension(500, 90));
	    
	    mainPane.add(topPane);
	    
		add(mainPane);
	}
}

@SuppressWarnings("serial")
class AnalysisMenuBar extends BaseSaveMenuBar {
	
	private JMenu contrastMenu;
	private JMenuItem deselectBehaviourBlockConditions;
	private JMenuItem deselectConditions;
	
	private Vector<String[]> sessionProfiles;
	private Vector<Integer> conditionSelection;
	private Vector<Integer> behaviorBlockConditionSelection;
	private JFrame parent;
	
	public AnalysisMenuBar(boolean isNpairsAnalysis, 
						   JFrame parent, 
						   Vector<Integer> conditionSelection, 
						   Vector<Integer> behaviourBlockConditionSelection,
						   Vector<String[]> sessionProfiles) {
		
		this(parent, 
			 conditionSelection, 
			 behaviourBlockConditionSelection, 
			 sessionProfiles);
		
		if (isNpairsAnalysis) {
			contrastMenu.setEnabled(false);
			deselectBehaviourBlockConditions.setEnabled(false);
		}
		
	}
	
	public AnalysisMenuBar(JFrame mainframe, 
			Vector<Integer> cs, 
			Vector<Integer> behaviorBlockCs, 
			Vector<String[]> sProfiles) {
		
		super(mainframe);
		
		sessionProfiles = sProfiles;
		conditionSelection = cs;
		behaviorBlockConditionSelection = behaviorBlockCs;
		parent = mainframe;
		
        // Build the deselect menu
		buildDeselectMenu(parent, 
						  conditionSelection, 
						  behaviorBlockConditionSelection, 
						  sessionProfiles);	
        
        // Build the contrast menu
        contrastMenu = new JMenu("Contrast");
        contrastMenu.setMnemonic('C');
        contrastMenu.getAccessibleContext().setAccessibleDescription("Contrast");
        add(contrastMenu, 2);
        
        JMenuItem openContrastsWindow = new JMenuItem("Open Contrast Window", 
        		new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Movie16.gif")));
        openContrastsWindow.setMnemonic('O');
        openContrastsWindow.getAccessibleContext().setAccessibleDescription("Create a contrasts file");
        
        openContrastsWindow.addActionListener(
        		new ContrastListener(sessionProfiles,conditionSelection,
        			((AnalysisFrame) parent).plsOptionPanel.contrastFilenameField));
        
        openContrastsWindow.setEnabled(true); 
        contrastMenu.add(openContrastsWindow);
        
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
		
		setFileFilter(new PlsAnalysisSetupFileFilter(), "_PLSAnalysisSetup.mat");
    }
	
	/**
	 * Disable/Enable the contrast menu item depending on the type of pls 
	 * option selected.
	 * @param val True if the menu item should be enabled.
	 */
	public void enableContrastMenu(boolean val){
		contrastMenu.setEnabled(val);
	}
	
	public void updateDeselectMenuConds(JFrame parent, Vector newConditionSelection, Vector <String[]> sessionProfiles) {
		deselectConditions.removeActionListener(deselectConditions.getActionListeners()[0]);
		deselectConditions.addActionListener(new DeselectConditionsListener(parent, newConditionSelection, sessionProfiles));
	}
	
	public void updateDeselectMenuBehavBlockConds(JFrame parent, Vector newBehavBlockConditionSelection, Vector <String[]> sessionProfiles) {
		deselectBehaviourBlockConditions.removeActionListener(deselectBehaviourBlockConditions.getActionListeners()[0]);
		deselectBehaviourBlockConditions.addActionListener(new DeselectConditionsListener(parent, newBehavBlockConditionSelection, sessionProfiles));
	}
	
	public void buildDeselectMenu(JFrame parent, Vector conditionSelection, Vector behaviorBlockConditionSelection,
				Vector<String[]> sessionProfiles) {
		JMenu deselectMenu = new JMenu("Deselect");
        deselectMenu.setMnemonic('D');
        deselectMenu.getAccessibleContext().setAccessibleDescription("Deselectable variables");
        add(deselectMenu, 1);
        
        deselectConditions = new JMenuItem(
        		"Deselect conditions (before loading behavior data)",
        		new ImageIcon(this.getClass().getResource(
        				"/toolbarButtonGraphics/general/Preferences16.gif")));
        deselectConditions.setMnemonic('C');
        deselectConditions.getAccessibleContext().setAccessibleDescription("Filter out conditions you don't want to be a part of the analysis");
        deselectConditions.addActionListener(new DeselectConditionsListener(parent, conditionSelection, sessionProfiles));
        deselectMenu.add(deselectConditions);
        
        deselectBehaviourBlockConditions = new JMenuItem("Deselect behavior block conditions for Multiblock PLS (after loading behavior data)", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Preferences16.gif")));
        deselectBehaviourBlockConditions.setMnemonic('B');
        deselectBehaviourBlockConditions.getAccessibleContext().setAccessibleDescription("Filter out conditions you don't want to be a part of the analysis");
        deselectBehaviourBlockConditions.addActionListener(new DeselectConditionsListener(parent, behaviorBlockConditionSelection, sessionProfiles));
        deselectBehaviourBlockConditions.setEnabled(false); // TODO: re-enable when multiblock works
        deselectMenu.add(deselectBehaviourBlockConditions);
	}
	
	public void clearFields() {
		AnalysisFrame frame = (AnalysisFrame) parent;
		
		// Clear the session file info for each group.
		SessionProfilesPanel sessionProfilesPanel = frame.sessionProfilePanel;
		DefaultTableModel model = sessionProfilesPanel.model;
		
		int numRows = model.getRowCount();
		for (int i = numRows - 1; i != -1; i--) {
			model.removeRow(i);
		}
		sessionProfilesPanel.sessionProfiles.removeAllElements();
		sessionProfilesPanel.conditionSelection.removeAllElements();
		sessionProfilesPanel.behaviorBlockConditionSelection.removeAllElements();
		
		// Reset the PLS analysis type info.
		PlsOptionPanel plsOptionPanel = frame.plsOptionPanel;
		
		plsOptionPanel.radioButtons[0].setSelected(true);
		
		// Clear the data file fields.
		plsOptionPanel.contrastFilenameField.setText("");
		plsOptionPanel.behaviorFilenameField.setText("");
		//plsOptionPanel.posthocFilenameField.setText("");
		
		// Reset the permutations/bootstrap info.
		PermBootPanel permBootPanel = frame.permBootPanel;
		
		permBootPanel.numPermsField.setText("0");
		permBootPanel.numBootsField.setText("0");
		permBootPanel.confidenceField.setText("95");
		
		// Clear the results file name.
		frame.resultsFilenameField.setText("");
	}
	
	@Override
	public void load() {
		AnalysisFrame frame = (AnalysisFrame) parent;

		// Get PLS setup parameters from the file.
		MLStructure plsSetupInfo = null;
		try {
			plsSetupInfo = (MLStructure) new NewMatFileReader(fileName).getContent().get("pls_setup_info");
		} catch (Exception e) {
    		JOptionPane.showMessageDialog(null, "PLS setup parameter file " + fileName 
    				+ "\ncould not be loaded.", "Error", JOptionPane.ERROR_MESSAGE);
    		return;
		}
		
		// Load the condition selection info.
		MLDouble conditionSelectionInfo = (MLDouble) plsSetupInfo.getField("cond_selection");
		int numConds = conditionSelectionInfo.getM();
		Vector<Integer> conditionSelection;
		
		conditionSelection = frame.sessionProfilePanel.getConditionSelection();
		conditionSelection.removeAllElements();
		conditionSelection.ensureCapacity(numConds);
		
		for (int i = 0; i < numConds; i++) {
			conditionSelection.add(conditionSelectionInfo.get(i).intValue());
		}
		
		// Load the behavior block condition selection info.
		MLDouble behavBlockConditionSelectionInfo = (MLDouble) plsSetupInfo.getField("behav_block_cond_selection");
		numConds = behavBlockConditionSelectionInfo.getM();
		Vector<Integer> behavBlockConditionSelection = new Vector<Integer>(numConds);
		for (int i = 0; i < numConds; i++) {
			behavBlockConditionSelection.add(behavBlockConditionSelectionInfo.get(i).intValue());
		}
		
		// Load the session profile info.
		MLStructure sessionFileInfo = (MLStructure) plsSetupInfo.getField("session_file_info");
		int numGroups = sessionFileInfo.getN();
		Vector<String[]> sessionProfiles;
		
		sessionProfiles = frame.sessionProfilePanel.getSessProfiles();
		sessionProfiles.removeAllElements();
		sessionProfiles.ensureCapacity(numGroups);
		
		for (int i = 0; i < numGroups; i++) {
			String[] currSessFiles = MLFuncs.MLCell1dRow2StrArray((MLCell)(sessionFileInfo.getField("session_files", i)));
			sessionProfiles.add(i, currSessFiles);
		}
		
		//Assumption: all session files have the same conditions
		String sessionFileName = sessionProfiles.get(0)[0];
			
		try {
			MLStructure sessProfStruct = (MLStructure) new NewMatFileReader(sessionFileName).
					getContent().get("session_info");
			String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
//			for (int i = 0; i < conditions.length; ++i) {
//				//TODO: read conditionSelection info in from 
//				System.out.println("Reading conditionSelection... ");
//				conditionSelection.add(new
//				}
		} catch (FileNotFoundException e) {
			JOptionPane.showMessageDialog(null, "Could not find PLS session file " + sessionFileName +   
					".", "Error", JOptionPane.ERROR_MESSAGE);
    		return;
		} catch (IOException e) {
			JOptionPane.showMessageDialog(null, "Could not load PLS session file " + sessionFileName + 
					".", "Error", JOptionPane.ERROR_MESSAGE);
		}
		
		SessionProfilesPanel sessionProfilesPanel = frame.sessionProfilePanel;
		sessionProfilesPanel.setSessProfiles(sessionProfiles);
		sessionProfilesPanel.setBehavBlockConditionSelection(behavBlockConditionSelection);
		
		updateDeselectMenuConds(frame, conditionSelection, sessionProfiles);
		updateDeselectMenuBehavBlockConds(frame, behavBlockConditionSelection, sessionProfiles);
		
		// Load the PLS analysis type info.
		PlsOptionPanel plsOptionPanel = frame.plsOptionPanel;
		boolean meanCenteringPLS = (((MLDouble) plsSetupInfo.getField("mean-centering_PLS")).get(0,0).intValue() == 1);
		boolean behaviorPLS = (((MLDouble) plsSetupInfo.getField("behavior_PLS")).get(0,0).intValue() == 1);
		boolean nonRotatedTaskPLS = (((MLDouble) plsSetupInfo.getField("non-rotated_task_PLS")).get(0,0).intValue() == 1);
		boolean multiblockPLS = (((MLDouble) plsSetupInfo.getField("multiblock_PLS")).get(0,0).intValue() == 1);
		boolean nonRotatedBehavPLS;
		
		if (plsSetupInfo.getField("non-rotated_behavior_PLS")!=	null) {
			nonRotatedBehavPLS = (((MLDouble) plsSetupInfo.getField("non-rotated_behavior_PLS")).get(0,0).intValue() == 1);
		} else {
			nonRotatedBehavPLS = false;
		}
				
		plsOptionPanel.radioButtons[0].setSelected(meanCenteringPLS);
		plsOptionPanel.radioButtons[1].setSelected(behaviorPLS);
		plsOptionPanel.radioButtons[2].setSelected(nonRotatedTaskPLS);
		plsOptionPanel.radioButtons[3].setSelected(multiblockPLS);
		plsOptionPanel.radioButtons[4].setSelected(nonRotatedBehavPLS);
		
		// Load the data file fields.
		String contrastFilename = ((MLChar) plsSetupInfo.getField("contrast_data_filename")).getString(0);
		plsOptionPanel.contrastFilenameField.setText(contrastFilename);

		String behaviorFilename = ((MLChar) plsSetupInfo.getField("behavior_data_filename")).getString(0);
		plsOptionPanel.behaviorFilenameField.setText(behaviorFilename);
		
		// Load the permutations/bootstrap info.
		PermBootPanel permBootPanel = frame.permBootPanel;

		String numPerms = ((MLChar) plsSetupInfo.getField("num_permutations")).getString(0);
		permBootPanel.numPermsField.setText(numPerms);

		String numBoots = ((MLChar) plsSetupInfo.getField("num_bootstraps")).getString(0);
		permBootPanel.numBootsField.setText(numBoots);

		String confidenceLevel = ((MLChar) plsSetupInfo.getField("confidence_level")).getString(0);
		permBootPanel.confidenceField.setText(confidenceLevel);
		
		// Load the results file name.
		String result = ((MLChar) plsSetupInfo.getField("results_filename")).getString(0);
		frame.resultsFilenameField.setText(result);
	}
	
	@Override
	public void save() {
		
		int maxNumSessions = 0;
		int numGroups = sessionProfiles.size();
		if (!(numGroups > 0)) {
			JOptionPane.showMessageDialog(null, "Error saving PLS setup file " + fileName + " - " + 
					"must include at least one session file.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		for (int i = 0; i < numGroups; i++) {
			int currNumSessions = sessionProfiles.get(i).length;
			maxNumSessions = Math.max(maxNumSessions, currNumSessions);
		}
		
		MLStructure plsSetupInfo = new MLStructure("pls_setup_info", new int[] {1, 1});
		
		// Save the session file info for each group.
		MLStructure sessionFileInfo = new MLStructure("session_file_info", new int[] {1, numGroups});
		for (int i = 0; i < numGroups; i++) {
			int currNumSessions = sessionProfiles.get(i).length;
			MLCell currSessionFiles = new MLCell("session_files" + i, new int[] {1, currNumSessions});
			for (int sf = 0; sf < currNumSessions; sf++) {
				currSessionFiles.set(new MLChar("session_file" + sf, sessionProfiles.get(i)[sf]), 0, sf);
			}
			sessionFileInfo.setField("session_files", currSessionFiles, i);
			sessionFileInfo.setField("grp_size", new MLDouble("grp_size" + i, new double[][]{{currNumSessions}}), i);
		}
		plsSetupInfo.setField("session_file_info", sessionFileInfo);
		
		// Save condition selection info.
		if (conditionSelection == null) {
			String sessProfFileName = sessionProfiles.get(0)[0];
			try {
				MLStructure sessProfStruct = (MLStructure) new NewMatFileReader(sessProfFileName).getContent().get("session_info");
				String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
				conditionSelection = new Vector<Integer>(conditions.length);
				for (int i = 0; i < conditions.length; i++) {
					conditionSelection.add(new Integer(1));
				}
			} catch (FileNotFoundException e) {
				JOptionPane.showMessageDialog(null, "Could not find PLS session file " + sessProfFileName +   
						".", "Error", JOptionPane.ERROR_MESSAGE);
	    		return;
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "Could not load PLS session file " + sessProfFileName + 
						".", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
		int numConditions = conditionSelection.size();
		MLDouble condSelect = new MLDouble("cond_selection", new int[]{numConditions, 1});
		for (int i = 0; i < numConditions; ++i) {
			condSelect.set(new Double(conditionSelection.get(i)), i);
		}
		plsSetupInfo.setField("cond_selection", condSelect);
		
		// Save condition selection info.
		if (behaviorBlockConditionSelection == null) {
			String sessProfFileName = sessionProfiles.get(0)[0];
			try {
				MLStructure sessProfStruct = (MLStructure) new NewMatFileReader(sessProfFileName).getContent().get("session_info");
				String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
				behaviorBlockConditionSelection = new Vector<Integer>(conditions.length);
				for (int i = 0; i < conditions.length; i++) {
					behaviorBlockConditionSelection.add(new Integer(1));
				}
			} catch (FileNotFoundException e) {
				JOptionPane.showMessageDialog(null, "Could not find PLS session file " + sessProfFileName +   
						".", "Error", JOptionPane.ERROR_MESSAGE);
	    		return;
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "Could not load PLS session file " + sessProfFileName + 
						".", "Error", JOptionPane.ERROR_MESSAGE);
			}
		}
		numConditions = behaviorBlockConditionSelection.size();
		condSelect = new MLDouble("behav_block_cond_selection", new int[]{numConditions, 1});
		for (int i = 0; i < numConditions; ++i) {
			condSelect.set(new Double(behaviorBlockConditionSelection.get(i)), i);
		}
		plsSetupInfo.setField("behav_block_cond_selection", condSelect);
		
		// Save PLS analysis type info.
		AnalysisFrame frame = (AnalysisFrame) parent;
		PlsOptionPanel plsOptionPanel = frame.plsOptionPanel;
		
		if (plsOptionPanel.radioButtons[0].isSelected()) {
			plsSetupInfo.setField("mean-centering_PLS", new MLDouble("mean-centering_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("mean-centering_PLS", new MLDouble("mean-centering_PLS", new double[][]{{0}}));
		}
		if (plsOptionPanel.radioButtons[1].isSelected()) {
			plsSetupInfo.setField("behavior_PLS", new MLDouble("behavior_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("behavior_PLS", new MLDouble("behavior_PLS", new double[][]{{0}}));
		}
		if (plsOptionPanel.radioButtons[2].isSelected()) {
			plsSetupInfo.setField("non-rotated_task_PLS", new MLDouble("non-rotated_task_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("non-rotated_task_PLS", new MLDouble("non-rotated_task_PLS", new double[][]{{0}}));
		}
		if (plsOptionPanel.radioButtons[3].isSelected()) {
			plsSetupInfo.setField("multiblock_PLS", new MLDouble("multiblock_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("multiblock_PLS", new MLDouble("multiblock_PLS", new double[][]{{0}}));
		}
		if (plsOptionPanel.radioButtons[4].isSelected()) {
			plsSetupInfo.setField("non-rotated_behavior_PLS", new MLDouble("non-rotated_behavior_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("non-rotated_behavior_PLS", new MLDouble("non-rotated_behavior_PLS", new double[][]{{0}}));
		}
		
		// Save data file fields.
		String contrastFilename = plsOptionPanel.contrastFilenameField.getText().trim();
		plsSetupInfo.setField("contrast_data_filename", new MLChar("contrast_data_filename", contrastFilename));
		
		String behaviorFilename = plsOptionPanel.behaviorFilenameField.getText().trim();
		plsSetupInfo.setField("behavior_data_filename", new MLChar("behavior_data_filename", behaviorFilename));
		
		// Save permutations/bootstrap info.
		PermBootPanel permBootPanel = frame.permBootPanel;
		
		String numPerms = permBootPanel.numPermsField.getText().trim();
		plsSetupInfo.setField("num_permutations", new MLChar("num_permutations", numPerms));
		
		String numBoots = permBootPanel.numBootsField.getText().trim();
		plsSetupInfo.setField("num_bootstraps", new MLChar("num_bootstraps", numBoots));
		
		String confidenceLevel = permBootPanel.confidenceField.getText().trim();
		plsSetupInfo.setField("confidence_level", new MLChar("confidence_level", confidenceLevel));
		
		// Save the results file name.
		String result = frame.resultsFilenameField.getText().trim();
		plsSetupInfo.setField("results_filename", new MLChar("results_filename", result));
	
		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(plsSetupInfo);
		try {
			new MatFileWriter(fileName, list);
		} catch (Exception e) {
			JOptionPane.showMessageDialog(null, "Could not save to PLS setup file " + fileName + ".", 
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
	}
	
}

final class ContrastListener implements ActionListener {
	
	private Vector<String[]> sessionProfiles = null;
	private Vector<Integer> conditionSelection = null;
	private JTextField cField;
	
	public ContrastListener(Vector<String[]> sessionProfiles, 
							Vector<Integer>  conditionSelection,
							JTextField contrastField) {
	
		this.sessionProfiles = sessionProfiles;
		this.conditionSelection = conditionSelection;
		cField = contrastField;
		
		
		
	}
	
//	public ContrastListener(Vector<String[]> sessionProfiles) {
//		this.sessionProfiles = sessionProfiles;
//	}
	
	public void actionPerformed(ActionEvent e) {
		
		
		new ContrastsFrame(sessionProfiles, conditionSelection, 
							cField.getText());
	}
}

final class DeselectConditionsListener implements ActionListener {
	
	private JFrame parent = null;
	
	private Vector conditionSelection = null;
	
	private Vector<String[]> sessionProfiles = null;
	
	public DeselectConditionsListener(JFrame parent, Vector conditionSelection, Vector<String[]> sessionProfiles) {
		this.parent = parent;
		this.conditionSelection = conditionSelection;
		this.sessionProfiles = sessionProfiles;
	}
	public void actionPerformed(ActionEvent e) {
		new DeselectConditionsFrame(parent, sessionProfiles, conditionSelection);
	}
}