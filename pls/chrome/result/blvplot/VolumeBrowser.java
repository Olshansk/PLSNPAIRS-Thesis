package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.border.LineBorder;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;

import pls.chrome.result.LoadedVolumesDialog;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")

//This class represents the VolumeBrowser Panel (Left hand side of the PLS Results frame) -f
public class VolumeBrowser extends JPanel implements ActionListener, TreeSelectionListener {
	//private JLabel mVolumeBrowserLabel = new JLabel("Volume Browser"); //never used -f
	
	private GeneralRepository mRepository = null;
	
	private ArrayList<String> mAllResultFiles = null;
	
	private JButton mAddFilesButton = new JButton("Add/Remove Result Files");
	private JTree mTree = null;
	private DefaultTreeModel mTreeModel = null;
	private JCheckBox mInvertCheckBox = new JCheckBox(); //checkbox for inverting cv values -f
	
	
	public VolumeBrowser(GeneralRepository repository) {
		mRepository = repository;
		
		initializeWidgetsAsTree();
	}
	
	private void initializeWidgetsAsTree() {
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		setBorder(new LineBorder(Color.DARK_GRAY, 1) );
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS) );
		
		// Make the label have a bold font
		//Font propFont = mVolumeBrowserLabel.getFont(); //does nothing -f
		//mVolumeBrowserLabel.setFont(new Font(propFont.getName(), Font.BOLD, propFont.getSize() + 1) ); //does nothing -f
//		add(mVolumeBrowserLabel);
		
		// Add the add/remove files button
		mAddFilesButton.setAlignmentX(JButton.LEFT_ALIGNMENT);
		add(mAddFilesButton);
		
		DefaultMutableTreeNode rootNode = new DefaultMutableTreeNode("root", true);
		
		mTreeModel = new DefaultTreeModel(rootNode);
		mTree = new JTree(mTreeModel);
                //allows only a single item to be selected at once.
		mTree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
		
		// A simple mouse listener for double clicks on leaf nodes
		MouseListener ml = new MouseAdapter() {
		     public void mousePressed(MouseEvent e) {
		         int selRow = mTree.getRowForLocation(e.getX(), e.getY());
		         TreePath selPath = mTree.getPathForLocation(e.getX(), e.getY());
		         if(selRow != -1) {
		        	 if(e.getClickCount() == 2) { //check that user selected a valid row with a double click -f
						//check that selected item in the tree browser was a leaf -f
						 if (selPath.getLastPathComponent() instanceof VolumeBrowserLeafNode) {
		        			 VolumeBrowserLeafNode node = (VolumeBrowserLeafNode)selPath.getLastPathComponent();

							 //reverse display value because we just double clicked the node.  -f
		            		 node.beingDisplayed = !node.beingDisplayed;
		            		 
		            		 boolean viewLv = node.beingDisplayed;
                                         
		            		 ResultModel model = mRepository.getGeneral(); //retrieves the currently selected model -f
		            		 int lv = model.getBrainData().getLv() + 1; 
		            		 String type = model.getSelectedDataType(); //"Average Z-Scored eigenimage etc..." -f
		            		 ArrayList<Integer> viewedLvs = model.getViewModel().getViewedLvs(type);

							 //we want to view the lv, so add it to the collection of viewed lvs.
							 //I'm not sure if the second condition is necessary since we already
							 //always remove the lv when we don't want to view the lv anymore. -f
		            		 if (viewLv && !viewedLvs.contains(lv) ) {
		            			 viewedLvs.add(lv);
		            			 Collections.sort(viewedLvs);
		            		 }
		            		 else {
		            			 viewedLvs.remove(viewedLvs.indexOf(lv) );
		            		 }
							
		            		 ResultsCommandManager.setViewedLvs(type, viewedLvs);
		        		 }
		        	 }
		         }
		     }
		};
		 
		mTree.addMouseListener(ml);
		
		VolumeBrowserCellRenderer renderer = new VolumeBrowserCellRenderer();
		mTree.setCellRenderer(renderer);
		
		refreshTree();
		
		JScrollPane scroller = new JScrollPane(mTree);
		
		scroller.setAlignmentX(JScrollPane.LEFT_ALIGNMENT);
		scroller.setPreferredSize(new Dimension(300, scroller.getPreferredSize().height));
		
		add(scroller);
		JPanel contextPanel = new JPanel();
		contextPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		contextPanel.setBorder(new LineBorder(Color.LIGHT_GRAY, 1) );
		
		add(mInvertCheckBox);
		
		mAddFilesButton.addActionListener(this);
	}
	
	public void selectLv() {
		mTree.removeTreeSelectionListener(this);
		
		String file = getListName(mRepository.getSelectedResultFile() );
		String type = mRepository.getGeneral().getSelectedDataType();
		int lvNum = mRepository.getGeneral().getBrainData().getLv();
		String lvString = mRepository.getGeneral().getVariableType() + " #" + (lvNum + 1);
		
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode)mTreeModel.getRoot();
		DefaultMutableTreeNode fileNode = findChildNode(rootNode, file);
		DefaultMutableTreeNode typeNode = findChildNode(fileNode, type);
		DefaultMutableTreeNode lvNode = findChildNode(typeNode, lvString);
		
		TreePath path = new TreePath(lvNode.getPath());
		mTree.getSelectionModel().setSelectionPath(path);
		
		mInvertCheckBox.removeActionListener(this);
		mInvertCheckBox.setSelected(mRepository.getGeneral().getBrainData().getColourScaleModel().isInverted(lvNum) );
		mInvertCheckBox.addActionListener(this);
		
		mTree.addTreeSelectionListener(this);
	}
	
	private DefaultMutableTreeNode findChildNode(DefaultMutableTreeNode node, String childName) {
		Enumeration<DefaultMutableTreeNode> children = node.children();
		
		DefaultMutableTreeNode child = null;
		
		while (children.hasMoreElements() ) {
			child = children.nextElement();
			
			if (childName.equals(child.toString()) ) {
				return child;
			}
		}
		
		return null;
	}

	public void actionPerformed(ActionEvent event) {
		if (event.getSource() == mAddFilesButton) {
			LoadedVolumesDialog.showDialog(this, mRepository);
		}
		else if (event.getSource() == mInvertCheckBox) {
			ResultsCommandManager.toggleInverted();
		}
	}
	
	private String getListName(String longName) {
		char seperator = File.separatorChar;
		return longName.substring(longName.lastIndexOf(seperator) + 1);
	}

	// TODO This function is actually called twice every time a node is selected
	// from the tree. This is very peculiar and should be fixed.
	// It seems to have its origins in two action listeners being present.
	// If the mTree action listener is removed, only one event happens.
	public void valueChanged(TreeSelectionEvent e) {
		mTree.removeTreeSelectionListener(this);
		
		TreePath path = e.getPath();
		
		Object[] pathItems = path.getPath();
		int pathLength = pathItems.length;

		// Item 0 is the root
		if (pathLength > 1) {
			//Last selected item. If no item has ever been selected item is last
			//Node in the tree and parent is null.
			DefaultMutableTreeNode fileNode = (DefaultMutableTreeNode)pathItems[1];

			int fileIndex;

			//If no file was select return. This function is normally called
			//when a node is selected but it can also be called when a file
			//is added (which means that no node was selected and so we
			//should return).
			if(fileNode.getParent() == null){
				mTree.addTreeSelectionListener(this);
				return;
			}else{
				//index of filenode relative to location within the parent.
				fileIndex = fileNode.getParent().getIndex(fileNode);
			}

			String selectedFile = mAllResultFiles.get(fileIndex);
			
			ResultsCommandManager.selectResultFile(selectedFile);

			if (pathLength > 2) {
				DefaultMutableTreeNode typeNode = (DefaultMutableTreeNode)pathItems[2];
				String selectedType = typeNode.toString();

				ResultsCommandManager.selectBrainData(selectedType);
				
				if (pathLength > 3) {
					// Get the lv that was selected for that data type
					int lv = mRepository.getGeneral().getBrainData(selectedType).getLv();
					
					DefaultMutableTreeNode lvNode = (DefaultMutableTreeNode)pathItems[3];
					lv = lvNode.getParent().getIndex(lvNode);
					
					ResultsCommandManager.selectLv(lv);
				}
			}
		}
		
		mTree.getSelectionModel().setSelectionPath(path);
		mTree.addTreeSelectionListener(this);
	}

	public void refreshTree() {
		mTree.removeTreeSelectionListener(this);
		mTree.setRootVisible(true);
		//mTree.collapseRow(0);
		
		mInvertCheckBox.setText("Negate values for this " + mRepository.getGeneral().getAbbrVariableType());
		mInvertCheckBox.setEnabled(true); 
		
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode)mTreeModel.getRoot();
		rootNode.removeAllChildren();
		
		mAllResultFiles = new ArrayList<String>(mRepository.getModels() );
		
		for (String modelName : mAllResultFiles) {
			DefaultMutableTreeNode modelNode = new DefaultMutableTreeNode(getListName(modelName) );
			ResultModel model = mRepository.getGeneral(modelName);
			
			for (String dataType : model.getBrainDataTypes() ) {
				DefaultMutableTreeNode typeNode = new DefaultMutableTreeNode(dataType);
				BrainData brainData = model.getBrainData(dataType);
				
				ArrayList<Integer> viewedLvs = model.getViewModel().getViewedLvs(dataType);
				
				for (int lv = 1; lv <= brainData.getNumLvs(); ++lv) {
					VolumeBrowserLeafNode lvNode = new VolumeBrowserLeafNode(model.getVariableType() + " #" + lv);
					lvNode.beingDisplayed = viewedLvs.contains(lv);
					typeNode.add(lvNode);
				}
				modelNode.add(typeNode);
			}
			rootNode.add(modelNode);
		}
		
		//mTree.expandRow(0);
		mTree.setRootVisible(false);
		//Set selected node to be the top most node.
		//mTree.getSelectionModel().get
		mTreeModel.reload();
		selectLv();
		mTree.addTreeSelectionListener(this);
	}

	public void updateInvertedLv() {
		boolean isInverted = mRepository.getGeneral().getBrainData().getColourScaleModel().isInverted();
		
		mInvertCheckBox.removeActionListener(this);
		mInvertCheckBox.setSelected(isInverted);
		mInvertCheckBox.addActionListener(this);
	}
}
