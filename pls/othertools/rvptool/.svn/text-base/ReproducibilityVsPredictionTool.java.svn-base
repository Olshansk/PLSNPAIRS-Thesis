package pls.othertools.rvptool;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.JTree;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.filechooser.FileFilter;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;

import pls.shared.ResultFileFilter;

@SuppressWarnings("serial")
/**
 * Reproducibility vs Prediction metaplot creation tool.
 */

public class ReproducibilityVsPredictionTool extends JFrame
		implements ActionListener, TreeSelectionListener {

//	public static void main(String[] args) {
//		ReproducibilityVsPredictionTool mainFrame =
//				new ReproducibilityVsPredictionTool();
//		mainFrame.setSize(1024, 768);
//		mainFrame.setVisible(true);
//	}

	public String unitType = "CV";
	protected String errorString = "";
	
	protected ArrayList<CurveGroup> mCurves;

	//private boolean combineWarnings = false;
	
	private JComboBox mCurveBox;
	private JTree mTree;
	private DefaultTreeModel mTreeModel;

	private JTextField mUnitField;
	private JTextField mNameField;
	private JTextField mFilenameField;
	private JTextField mVarField;
	private JComboBox mPredTypeList;
	private String[] mPredTypeStrings = { "Posterior Probability", 
	 									  "Percent Accuracy" };
	protected String predType = mPredTypeStrings[0];

	private JFileChooser jfc = new JFileChooser(".");
	private JFileChooser pBrowser = new JFileChooser(".");
	
	private JButton generatePlot = new JButton("Generate Plot");
	
	public ReproducibilityVsPredictionTool() {
		super("Reproducibility vs Prediction Plot Tool");
		setupGui();
	}

	protected ReproducibilityVsPredictionTool(String title, String unitType) {
		super(title);
		this.unitType = unitType;
		setupGui();
	}

	private void setupGui(){
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jfc.addChoosableFileFilter(new FileFilter(){

			@Override
			public boolean accept(File file) {
				if(file.isDirectory() || 
						file.getAbsolutePath().endsWith(".txt")){
					return true;
				}
				return false;
			}

			@Override
			public String getDescription() {
				return ".txt filter";
			}
			
		});
		
		ResultFileFilter rFF = new ResultFileFilter(){
			@Override
			public String getDescription(){
				return "Select a file in the desired dataset to " +
						"determine the general filename";
			}
		};
		pBrowser.addChoosableFileFilter(rFF);
		
		setupMenu();
		setupLeftTree();
		setupMainPanel();

		generatePlot.setActionCommand("Plot");
		generatePlot.addActionListener(this);
		getContentPane().add(generatePlot, BorderLayout.SOUTH);

		setSize(1024,480);
		setVisible(true);
		mCurves = new ArrayList<CurveGroup>();
		addNewCurve(); //add a new curve so the user can enter data in.
	}
	
	private void setupMenu() {
		JMenuBar menubar = new JMenuBar();
		
		JMenu fileMenu = new JMenu("File");
		JMenuItem newItem = new JMenuItem("Clear Curves");
		newItem.setActionCommand("New");
		newItem.addActionListener(this);
		JMenuItem load = new JMenuItem("Load");
		load.setActionCommand("Load");
		load.addActionListener(this);
		JMenuItem save = new JMenuItem("Save");
		save.setActionCommand("Save");
		save.addActionListener(this);
		JMenuItem exit = new JMenuItem("Exit");
		exit.setActionCommand("Exit");
		exit.addActionListener(this);

		fileMenu.add(newItem);
		fileMenu.add(load);
		fileMenu.add(save);
		fileMenu.add(exit);
		
		menubar.add(fileMenu);
		
		setJMenuBar(menubar);
	}
	
	private void setupLeftTree() {
		
		
		DefaultMutableTreeNode rootNode = new DefaultMutableTreeNode("root", true);
		mTreeModel = new DefaultTreeModel(rootNode);
		mTree = new JTree(mTreeModel);
//		mTree.setBorder(new LineBorder(Color.BLUE, 3));
		mTree.setRootVisible(false);
		mTree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
		mTree.addTreeSelectionListener(this);
		
		JScrollPane scroller = new JScrollPane(mTree);
		scroller.setPreferredSize(new Dimension(300,
				scroller.getPreferredSize().height));
		getContentPane().add(scroller, BorderLayout.WEST);
	}
	
	private void setupMainPanel() {		
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS) );
//		panel.setBorder(new LineBorder(Color.RED, 3));
		
		// Buttons to add or remove a group; panel to edit plot type
		JPanel buttonAndDropDownPanel = new JPanel();
		buttonAndDropDownPanel.setLayout(new BoxLayout(buttonAndDropDownPanel, BoxLayout.LINE_AXIS) );
		
		JButton removeThisCurve = new JButton("Remove This Curve");
		removeThisCurve.setActionCommand("Remove");
		removeThisCurve.addActionListener(this);
		
		JButton addNewCurve = new JButton("Add New Curve");
		addNewCurve.setActionCommand("Add");
		addNewCurve.addActionListener(this);
		
		buttonAndDropDownPanel.add(addNewCurve);
		buttonAndDropDownPanel.add(removeThisCurve);

		
		// The panel to edit the plot type 
		JPanel predTypePanel = new JPanel();
		predTypePanel.setLayout(new BoxLayout(predTypePanel, BoxLayout.LINE_AXIS));
		predTypePanel.add(new JLabel("Prediction Type to Plot: "));
		mPredTypeList = new JComboBox(mPredTypeStrings);
		mPredTypeList.setMaximumSize(new Dimension(
				mPredTypeList.getMaximumSize().width,
				mPredTypeList.getPreferredSize().height));
		mPredTypeList.setSelectedIndex(0);
		mPredTypeList.setActionCommand("Change type");
		mPredTypeList.addActionListener(this);
		predTypePanel.add(mPredTypeList);
		predTypePanel.setAlignmentX(LEFT_ALIGNMENT);
		
		buttonAndDropDownPanel.add(predTypePanel);
		buttonAndDropDownPanel.setAlignmentX(LEFT_ALIGNMENT);
		panel.add(buttonAndDropDownPanel);
		
		
		// Empty panel between button/drop-down options
		// and curve options to improve layout aesthetics
		JPanel emptyPanel = new JPanel();
		emptyPanel.setLayout(new BoxLayout(emptyPanel, BoxLayout.LINE_AXIS) );
		JLabel emptyLabel = new JLabel("");
		emptyLabel.setBorder(BorderFactory.createEmptyBorder(10,20,10,20));
		emptyPanel.add(emptyLabel);
		emptyPanel.setAlignmentX(LEFT_ALIGNMENT);
		panel.add(emptyPanel);
		
		// The panel to select a curve
		JPanel curvePanel = new JPanel();
		curvePanel.setLayout(new BoxLayout(curvePanel, BoxLayout.LINE_AXIS) );
		curvePanel.add(new JLabel("Browse available curves: ") );
		mCurveBox = new JComboBox();
		mCurveBox.setMaximumSize(new Dimension(
				mCurveBox.getMaximumSize().width,
				mCurveBox.getPreferredSize().height));
		mCurveBox.setActionCommand("Select");
		mCurveBox.addActionListener(this);
		curvePanel.add(mCurveBox);
		
		curvePanel.setAlignmentX(LEFT_ALIGNMENT);
		panel.add(curvePanel);
		
		// The panel to edit the curve's name
		JPanel namePanel = new JPanel();
		namePanel.setLayout(new BoxLayout(namePanel, BoxLayout.LINE_AXIS) );
		namePanel.add(new JLabel("Curve Label: ") );
		mNameField = new JTextField();
		mNameField.setMaximumSize(new Dimension(
				mNameField.getMaximumSize().width,
				mNameField.getPreferredSize().height));
		namePanel.add(mNameField);
		
		namePanel.setAlignmentX(LEFT_ALIGNMENT);
		panel.add(namePanel);
		
		// The panel to edit the curve's filename
		JPanel filenamePanel = new JPanel();
		JButton pathBrowser = new JButton("Select dataset");
		
		pathBrowser.setActionCommand("browsePath");
		pathBrowser.addActionListener(this);
		
		filenamePanel.setLayout(new BoxLayout(filenamePanel, BoxLayout.LINE_AXIS) );
		filenamePanel.add(new JLabel("General Filename: ") );
		mFilenameField = new JTextField();
		mFilenameField.setMaximumSize(new Dimension(
				mFilenameField.getMaximumSize().width,
				mFilenameField.getPreferredSize().height));
		filenamePanel.add(mFilenameField);
		filenamePanel.add(pathBrowser);
		
		filenamePanel.setAlignmentX(LEFT_ALIGNMENT);
		panel.add(filenamePanel);
		
		// The panel to edit the group's unitType (e.g cv,session, etc).
		JPanel unitPanel = new JPanel();
		unitPanel.setLayout(new BoxLayout(unitPanel, BoxLayout.LINE_AXIS) );
		unitPanel.add(new JLabel(unitType + "#: ") );
		mUnitField = new JTextField();
		mUnitField.setMaximumSize(new Dimension(mUnitField.getMaximumSize().width,
				                           mUnitField.getPreferredSize().height));
		unitPanel.add(mUnitField);
		unitPanel.setAlignmentX(LEFT_ALIGNMENT);
		
		panel.add(unitPanel);
		
		// The panel to edit the variable
		JPanel varPanel = new JPanel();
		varPanel.setLayout(new BoxLayout(varPanel, BoxLayout.LINE_AXIS) );
		varPanel.add(new JLabel("PC range: ") );
		mVarField = new JTextField();
		mVarField.setMaximumSize(new Dimension(mVarField.getMaximumSize().width, 
				                          mVarField.getPreferredSize().height));
		varPanel.add(mVarField);
		varPanel.setAlignmentX(LEFT_ALIGNMENT);
		
		panel.add(varPanel);
		
		JButton applyChanges = new JButton("Apply");
		applyChanges.setActionCommand("Apply");
		applyChanges.addActionListener(this);
				
		panel.add(applyChanges);
		getContentPane().add(panel, BorderLayout.CENTER);
		
		//disable the generate plot button until we are ready to plot
		generatePlot.setEnabled(false);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		
		if (e.getActionCommand().equals("New") ) {
			clear();
		}
		else if (e.getActionCommand().equals("Load") ) {
			
			int option = jfc.showOpenDialog(this);
			
			if (option == JFileChooser.APPROVE_OPTION) {
				loadFile(jfc.getSelectedFile() );
			}
			
		} else if (e.getActionCommand().equals("Save") ) {
			
			int option = jfc.showSaveDialog(this);
			
			if (option == JFileChooser.APPROVE_OPTION) {
				saveFile(jfc.getSelectedFile(),
						 jfc.getFileFilter().getDescription()); 
			}
		} else if(e.getActionCommand().equals("Exit")){
			dispose();
		} else if (e.getActionCommand().equals("Add") ) {
			if(applyChange()){
				addNewCurve();
			}
		} else if (e.getActionCommand().equals("Remove") ) {
			removeCurve();
		} else if (e.getActionCommand().equals("Select") ) {
			if (mCurveBox.getItemCount() > 0 &&
				mCurveBox.getSelectedItem() != null) {

				String groupName = mCurveBox.getSelectedItem().toString();
				
				selectCurve(groupName);
			}
		} else if (e.getActionCommand().equals("Apply")) {
			applyChange();	
		} else if (e.getActionCommand().equals("Change type")) {
			predType = mPredTypeStrings[mPredTypeList.getSelectedIndex()];
		} else if (e.getActionCommand().equals("Plot") ) {
			new RvpPlot(mCurves, predType);
		} else if (e.getActionCommand().equals("browsePath")){
			applyBrowsePath();			
		}
		
	}

	private void applyBrowsePath(){
		
		
		if(pBrowser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION){
			File f = pBrowser.getSelectedFile();
			String path = f.getAbsolutePath();
			String generalPath = path.replaceFirst("\\d+pc","\\$pc");
			
			//Nothing was replaced so we loaded a bad file.
			if(generalPath.equals(path)){
				String message = "Selected file must be generalizable" +
						" and must be of the form: 'name_#pc_extension'";
				
				JOptionPane.showMessageDialog(this, message, "Bad file",
						JOptionPane.INFORMATION_MESSAGE);
				
				return;
			}
			mFilenameField.setText(generalPath);
		}
	}
	/**
	 * Respond from a click on the remove curve button by removing the selected
	 * curve.
	 */
	private void removeCurve() {
		if (mCurves.size() > 0) {
			String curveName = mCurveBox.getSelectedItem().toString();

			CurveGroup theGroup = null;

			for (CurveGroup group : mCurves) {
				if (group.getLabel().equals(curveName)) {
					theGroup = group;
					break;
				}
			}

			mCurves.remove(theGroup);

			//ends up calling valueChanged listener as well.
			removeCurve(theGroup);
			
			//If we have removed all groups then go ahead and create a
			//blank group just so the user has something to enter data into.
			if(mCurves.size() == 0){
				addNewCurve();
				generatePlot.setEnabled(false);
			}
		}
	}

	/**
	 * Respond from a click on the add curve button by adding the selected
	 * curve.
	 */
	private String addNewCurve(){
		String curveName = "New Curve";
		int suffix = 0;

		String theName = "";


		boolean itsOk = false;
		while (!itsOk) {
			itsOk = true;
			theName = curveName + ((suffix > 0) ? suffix : "");
			for (CurveGroup group : mCurves) {
				if (group.getLabel().equals(theName)) {
					++suffix;
					itsOk = false;
					break;
				}
			}

		}

		CurveGroup newGroup = new CurveGroup(theName, "", 1, "");
		addNode(newGroup);
		mCurves.add(newGroup);
		mCurveBox.addItem(theName);
		// Have it select the group we just added
		mCurveBox.setSelectedIndex(mCurveBox.getItemCount() - 1);
		
		return theName;
	}

	/**
	 * Respond from a click on the apply changes button by saving changes
	 * made so far by the user.
	 */
	private boolean applyChange(){
		String curveName = mCurveBox.getSelectedItem().toString(); 
		CurveGroup theGroup = null;

		for (CurveGroup group : mCurves) {
			if (group.getLabel().equals(curveName) ) {
				theGroup = group;
				break;
			}
		}
		
		boolean valid = validateInput(theGroup);

		if (valid) {
			DefaultMutableTreeNode node = findNode(theGroup);
			theGroup.setLabel(mNameField.getText().trim());
			theGroup.setFilename(mFilenameField.getText().trim());
			theGroup.setCurveUnit(Integer.parseInt(mUnitField.getText()));
			theGroup.setNumbers(mVarField.getText());
			updateNode(node, theGroup);

			mCurveBox.removeActionListener(this);
			int index = mCurveBox.getSelectedIndex();
			mCurveBox.removeItemAt(index);
			mCurveBox.insertItemAt(theGroup.getLabel(), index);
			mCurveBox.setSelectedIndex(index);
			mCurveBox.addActionListener(this);
			generatePlot.setEnabled(true);
		}
		return valid;
	}
	
	private void removeCurve(CurveGroup theGroup) {
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode)mTreeModel.getRoot();
		DefaultMutableTreeNode node = findNode(theGroup);
		
		rootNode.remove(node);
		
		mTreeModel.reload();

		mCurveBox.removeItem(theGroup.getLabel());
	}

	private void selectCurve(String curveName) {
		mCurveBox.removeActionListener(this);
		mTree.removeTreeSelectionListener(this);
		

		mCurveBox.setSelectedItem(curveName);
		
		CurveGroup theGroup = null;
		
		for (CurveGroup group : mCurves) {
			if (group.getLabel().equals(curveName) ) {
				theGroup = group;
				break;
			}
		}
		
		if (theGroup != null) {
			mNameField.setText(theGroup.getLabel() );
			mFilenameField.setText(theGroup.getFilename() );
			mUnitField.setText(Integer.toString(theGroup.getCurveUnit() ) );
			mVarField.setText(theGroup.getNumbers() );
			
			DefaultMutableTreeNode node = findNode(theGroup);

			if (mTree.getSelectionPath() != null) {
				Object[] pathItems = mTree.getSelectionPath().getPath();
				
				if (pathItems.length <= 1 || !pathItems[1].equals(node)) {
					mTree.setSelectionPath(new TreePath(node.getPath() ) );
				}
			} else if (node != null) {
				mTree.setSelectionPath(new TreePath(node.getPath() ) );
			}
		}
		
		
		mTree.addTreeSelectionListener(this);
		mCurveBox.addActionListener(this); 
	}

	/**
	 * validate the current fields the user has modified when attempting to 
	 * "apply changes" to a particular curve. examine the gui fields to 
	 * determine if the user has entered anything erroneous.
	 * @param curve a data structure holding the curve we are attempting to
	 * modify.
	 * @return true if ok, false otherwise.
	 */
	private boolean validateInput(CurveGroup group) {
		// Cannot allow duplicate labels
		String curveName = group.getLabel();
		String newGName = mNameField.getText().trim();
		String generalFileName = mFilenameField.getText().trim();
		String error;
		
		String oldFileName = group.getFilename();
		String oldLabel = group.getLabel();
		String oldNumbers = group.getNumbers();
		int oldUnit = group.getCurveUnit();
		
		boolean result;
		
		if(!newGName.equals(curveName)){

			for (int i = 0; i < mCurveBox.getItemCount(); ++i) {
				if (newGName.equals(mCurveBox.getItemAt(i).toString())) {
					error = "That curve label is already " +	"in use.";
					
					JOptionPane.showMessageDialog(this,
							error,"Label in use.", JOptionPane.OK_OPTION);
					return false;
				}
			}
		}
		
		// Filename must contain a wildcard ($)
		if (!generalFileName.contains("$") ) {
			error = "The general filename must contain a wildcard " +
			"($) character.";
			
			JOptionPane.showMessageDialog(this,
					error, "No wildcard in filename.",
					JOptionPane.OK_OPTION);
			return false;
		}

		// UnitType must be a number
		try {
			Integer val = Integer.parseInt(mUnitField.getText() );
			if(val < 1){
				error = "The " + unitType + " number must be a " +
						"valid integer > 0";
				
				JOptionPane.showMessageDialog(this, 
						error,"Invalid " + unitType +".", 
						JOptionPane.OK_OPTION);
				return false;
			}
		} catch (NumberFormatException nfex) {
			error = "The " + unitType + " number must be an integer.";
			JOptionPane.showMessageDialog(this, 
					error,"Invalid " + unitType +".", JOptionPane.OK_OPTION);
			return false;
		}
		
		//Check whether the specified files exist and that their curve units
		//are correct.
		
		//Set the group to have the new settings so we can test if they are 
		//correct.
		group.setFilename(generalFileName);
		group.setLabel(newGName);
		group.setCurveUnit(Integer.parseInt(mUnitField.getText()));
		group.setNumbers(mVarField.getText());
				
		result = validateExistence(group);
		
		if(!result){
			JOptionPane.showMessageDialog(this, errorString, 
					"Errors", JOptionPane.OK_OPTION);
		}
		
		//Reset the settings.
		group.setFilename(oldFileName);
		group.setLabel(oldLabel);
		group.setCurveUnit(oldUnit);
		group.setNumbers(oldNumbers);
		return result;
	}

	/**
	 * Validate whether the passed in matlab file exists
	 * @param group Information about the curve we want to plot from this 
	 * result file.
	 * @param checkFile A file object constructed from the absolute path
	 * of the result file.
	 * @return true if the file validates fine, false otherwise.
	 */
	private boolean validateMatFile(CurveGroup group, File checkFile){
//		int cu = group.getCurveUnit();
//		String fileName = checkFile.getAbsolutePath();
		String error = "";
		
		if(!checkFile.canRead()){
			error = "The file " + checkFile.getAbsolutePath() + " either" +
			" does not exist or is unreadable.";
			return Error(error);
		}
		return true;
		
		//Checking for CV existence was too slow and so this idea was scrapped.
		//The user is warned of a bad cv number when attempting to plot however.
/*		try {
			Map<String, MLArray>resultInfo;
			MLArray npairs_result;
			MLStructure struct_npairs_result;
			double[][] reprod_cc;
			
			resultInfo = new NewMatFileReader(fileName,
					new MatFileFilter(new String[]{"npairs_result",
													"reprod_cc"}))
												   .getContent();
			
			npairs_result = resultInfo.get("npairs_result");
			struct_npairs_result = (MLStructure) npairs_result;
			
			reprod_cc = ((MLDouble) struct_npairs_result
					.getField("reprod_cc")).getArray();
			
			if(reprod_cc[0].length < cu){
				error = "The file " + fileName + " does not contain"
				+ " the cv number: " + cu;
				
				return Error(error);
			}
			return true; //cv number exists.
		} 
		catch (IOException e) {
			error = "Error reading file " +	fileName;
			return Error(error);
		}*/
	}
	
	/**
	 * Set the global error string. This function is used in validateCCPPFile
	 * and validateMatFile so and errors can be aggregated and returned to the 
	 * calling function.
	 * @param error the error we want to log.
	 * @return always returns false.
	 */
	protected boolean Error(String error){
		if(errorString == null){
			errorString = error;
		}else{
			errorString += '\n' + error;
		}
		return false;
	}
	/**
	 * Verify whether the .CVA.SUMM.CC and .CVA.SUMM.PP.ppTruePriors exist
	 * as extensions of the file pointed at by 'fileName'. Also check whether
	 * the curve unit (cv/session/etc) is valid as specified in the 
	 * .CVA.SUMM.CC file.
	 * @param group Information about the curve we are currently validating.
	 * @param fileName The absolute location of where the data for this curve
	 * should be found.
	 * @return true if the file validates fine, false otherwise.
	 */
	private boolean validateCCPPFile(CurveGroup group, String fileName){
		File ccFile = new File(fileName + ".CVA.SUMM.CC");
		File ppFile = new File(fileName + ".CVA.SUMM.PP.ppTruePriors");
		String warning = "";
		String error = "";
		boolean ccRead = ccFile.canRead();
		boolean ppRead = ppFile.canRead();
		
		warning += !ccRead ? "Nonexistent/unreadable file "
				+ ccFile.getAbsolutePath() +"\n": "";
		warning += !ppRead ? "Nonexistent/unreadable file "
				+ ppFile.getAbsolutePath() +"\n": "";
			
		if(warning.equals("")){ //everything ok.
			return true;
			//Loading a matlab file to check for the cv was too slow so
			//we are disabling it here also for consistency.
			/*try{
				BufferedReader br = new BufferedReader(new FileReader(ccFile));
				String line = br.readLine();
				
				// These are the dimensions
				if (line != null) {
					String[] dims = line.split(" ");
					if(Integer.parseInt(dims[0]) < group.getCurveUnit()){
						error = "The file " + fileName + "does not contain"
						+ "the cv number: " + group.getCurveUnit();
						return Error(error);
					}
					return true; //cv was there.
				}
				error = "First line of CC file "
					+"must contain the dimensions and must not be " +
					"blank";
				return Error(error);
				
			}catch(IOException e){
				e.printStackTrace();
				error = "Error reading file " +	ccFile.getAbsolutePath();
				return Error(error);
			}*/
			
		}
		return Error(warning);
	}
	
	/**
	 * Validate that the necessary files for this curve exist 
	 * @param group a class holding information about the particular curve
	 * we want to plot.
	 * @return true if all necessary files exist and the curve unit is valid
	 * for all files. false otherwise. 
	 */
	protected boolean validateExistence(CurveGroup group){
		String generalFileName = group.getFilename();
		boolean error = false;
		
		errorString = null; //reset error string.
		// Check file existence.
		for(String number : group.getParsedNumbers()){
			String fileName = generalFileName.replace("$",number);
			File checkFile = new File(fileName);
			
			if(fileName.endsWith("NPAIRSJresult.mat")){
				//Check that validation did not fail.
				if(!validateMatFile(group, checkFile)){
					error = true;
				}
			}
			else{
				if(!validateCCPPFile(group,fileName)){
					error = true;
				}
			}
		}	
		return !error;
	}
	
	private DefaultMutableTreeNode findNode(CurveGroup group) {
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode)mTreeModel.getRoot();
		Enumeration<?> children = rootNode.children();
		
		while(children.hasMoreElements() ) {
			DefaultMutableTreeNode child = (DefaultMutableTreeNode)children.nextElement();
			
			String nodeName = child.getUserObject().toString();
			
			if (group.getLabel().equals(nodeName) ) {
				return child;
			}
		}
		
		return null;
	}
	
	private void updateNode(DefaultMutableTreeNode node, CurveGroup group) {
		if (node != null) {
			node.setUserObject(group.getLabel() );
			node.removeAllChildren();
			
			char seperator = File.separatorChar;
			String shortFilename = group.getFilename().substring(
					group.getFilename().lastIndexOf(seperator) + 1);
						
			node.add(new DefaultMutableTreeNode(unitType + "#" + group.getCurveUnit() ) );
			
			DefaultMutableTreeNode rangeNode = new DefaultMutableTreeNode("Files to use: ");
			ArrayList<String> numbers = group.getParsedNumbers();
			
			for (String i : numbers) {
				String currentFilename = shortFilename.replace("$", i);
				rangeNode.add(new DefaultMutableTreeNode(currentFilename) );
			}
			
			node.add(rangeNode);
			
			mTreeModel.reload(node);
		}
	}
	
	/**
	 * Load a saved text file containing information about result files to 
	 * plot.
	 * @param file the file in question.
	 */
	private void loadFile(File file) {
		mCurveBox.removeActionListener(this);
		mTree.removeTreeSelectionListener(this);
		
		// collect errors encountered in a single jpanel.
		String error = "";
				
		/*Apparently creating a Hashmap is faster than creating a Hashset
		 *if you just need set functionality but not a set implementation.
		 */
		//Map<String,Boolean> loadedGroups = new HashMap<String,Boolean>(10);
				
		HashSet<String> loadedGroups = new HashSet<String>(mCurves.size());
		for(CurveGroup g : mCurves){
			loadedGroups.add(g.getLabel());
		}
				
		//clear(); //reset and clear all groups since we are loading a file.
		
		int lineNum = 0;
		try {
			BufferedReader br = new BufferedReader(new FileReader(file) );
			
			try {
				String s = br.readLine();

				//for each line parsed attempt to create an rvp group.
				while (s != null) {
					lineNum++;
					
					if(s.equals("")){
						s = br.readLine();
						continue;
					}
					
					String[] items = s.split(","); // use ',' as a delimiter.
					
					if (items.length > 2) {
						String numbersString = "";

						//retrieve the result file numbers
						for (int i = 3; i < items.length; ++i) {
							numbersString += items[i];
							if (i != items.length -1) {
								numbersString += ",";
							}
						}
						
						//Check that we are using the correct unit type
						if(unitType.equals("Split object")){unitType = "SO";}
						if(!items[2].matches(unitType+"\\d+")){
							error+= "Error on line: " + lineNum 
							+ "\nFound " + items[2] +". Expecting " + 
							unitType + " unit type.\n";
							s = br.readLine();
							continue;
						}
						if(unitType.equals("SO")){unitType = "Split object";}
						
						//Check that the curve unit is a valid number.
						int unitNum;
						try {
							unitNum = Integer.parseInt(items[2].substring(2) );
							if(unitNum < 1){
								error += "Error on line: " + lineNum 
								+ "\nThe " + unitType.toLowerCase() + " must be an " +
										"integer > 0\n";
								s = br.readLine();
								continue;
							}
						} catch (NumberFormatException nfex) {
							error += "Error on line: " + lineNum
									+ "\nThe " + unitType
									+ " number must be an integer.\n";
							s = br.readLine();
							continue;
						}
						
						//Check that we aren't adding a duplicate group.
						items[0] = items[0].trim();
						if(loadedGroups.contains(items[0])){
							error += "Error on line: " + lineNum
							+ "\nDuplicate group (" + items[0] + ") found\n";
							s = br.readLine();
							continue;
						}
						
						//Check that the file name as a wild card character.
						items[1] = items[1].trim();
						if(!items[1].contains("$")){
							error += "Error on line: " + lineNum
							+ "\nThe general filename must contain a wildcard " +
							"($) character.";
							s = br.readLine();
							continue;
						}
						
						loadedGroups.add(items[0]);
						
						CurveGroup newGroup = new CurveGroup(items[0], //name
								                         items[1], //path
														 unitNum,
														 numbersString);
						
						//verify the group (file exists).
						if(!validateExistence(newGroup)){
							error += "Error on line: " + lineNum 
							+ "\n" + errorString + "\n";
							s = br.readLine();
							continue;
						}
						
						mCurves.add(newGroup);

						//add the group to the file selection tree.
						addNode(newGroup);

						mCurveBox.addItem(newGroup.getLabel() );
						s = br.readLine();
					} else {
						error += "Error on line: " + lineNum
								+ "\nNot enough information to be useful.\n";
						s = br.readLine();
					}
				}
			} catch (IOException ioex) {
				error += "Something really unexpected happened when " +
						"opening file " + file.getName() + "\n";
			}
		} catch (FileNotFoundException fnfex) {
			error += "Could not open file " + file.getName() + "\n";
		}
		
		if (!error.equals("") ) {
			JFrame errorFrame = new JFrame("Error log");
			JTextPane errorPane = new JTextPane();
			errorPane.setText(error);
			errorFrame.add(new JScrollPane(errorPane) );
			
			errorFrame.setVisible(true);
			errorFrame.setSize(640, 480);
		}
		
		mTree.addTreeSelectionListener(this);
		mCurveBox.addActionListener(this);
		
		if (mCurveBox.getItemCount() > 0) {
			selectCurve(mCurveBox.getItemAt(0).toString() );
		}
		
		//If we successfully added new curves remove the 'new curve'
		//which the user uses to enter data into.
		if(mCurves.size() > 1){
			clearNewGroup();
			generatePlot.setEnabled(true);
		}
	}
	
	/**
	 * Save the group information held in the rvp metaplot tool.
	 * This action happens when the user hits File->save.
	 * @param file The chosen file path.
	 * @param extDesc The description of the chosen file filter.
	 */
	protected void saveFile(File file, String extDesc) {
		
		//Append the appropriate extension
		if(extDesc.equals(".txt filter")){
			String absPath = file.getAbsolutePath();
			if(!absPath.endsWith(".txt")){
				file = new File(absPath+".txt");
			}
		}
		
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(file) );
			
			for(CurveGroup group : mCurves) {
				String line = group.getLabel() + ","
						    + group.getFilename() + "," + "CV"
							+ group.getCurveUnit() + ","
							+ group.getNumbers();
				bw.write(line);
				bw.newLine();
			}
			
			bw.close();
		} catch (IOException e) {
			String errorString = "Something really unexpected happened " +
					"when opening file " + file.getName() + "\n";
			JOptionPane.showMessageDialog(this, errorString, "IOError",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	/**
	 * Adds an rvp group to the selectable file tree.
	 * @param group the group in question.
	 */
	private void addNode(CurveGroup group) {
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode)mTreeModel.getRoot();
		DefaultMutableTreeNode node = new DefaultMutableTreeNode(group.getLabel() );

		char seperator = File.separatorChar;
		
		//filename of the group's result file with prefix removed.
		String shortFilename = group.getFilename().substring(
				group.getFilename().lastIndexOf(seperator) + 1);

		rootNode.add(node);

		node.add(new DefaultMutableTreeNode(unitType + "#" + group.getCurveUnit() ) );
		
		DefaultMutableTreeNode rangeNode = new DefaultMutableTreeNode("Files to use: ");
		ArrayList<String> numbers = group.getParsedNumbers();
		
		for (String i : numbers) {
			String currentFilename = shortFilename.replace("$", i);
			rangeNode.add(new DefaultMutableTreeNode(currentFilename) );
		}
		
		node.add(rangeNode);
		
		mTreeModel.reload();
		
		mTree.setSelectionPath(new TreePath(node.getPath() ) );
	}

	@Override
	public void valueChanged(TreeSelectionEvent e) {
		TreePath path = e.getPath();
		
		Object[] pathItems = path.getPath();
		
		// Item 0 is the root
		if (pathItems.length > 1) {
			String group = pathItems[1].toString();
			
			selectCurve(group);
		}
	}
	
	/**
	 * When Loading groups in from a file we no longer need the default
	 * 'New Curve' to enter data into so get rid of it.
	 */
	private void clearNewGroup(){
		CurveGroup defaultGroup = null;
		for(CurveGroup group : mCurves){
			if(group.getLabel().equals("New Curve")){
				defaultGroup = group;
				removeCurve(group);
			}
		}
		if(defaultGroup != null) mCurves.remove(defaultGroup);
	}
	
	private void clear() {
		for (CurveGroup group : mCurves) {
			removeCurve(group);
		}
		
		mCurves.clear();
		addNewCurve();
		generatePlot.setEnabled(false);
	}
}
