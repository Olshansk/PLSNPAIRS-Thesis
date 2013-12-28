package pls.chrome.analysis;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.Vector;

import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.filechooser.FileFilter;
import javax.swing.table.DefaultTableModel;

import pls.shared.NpairsBlockSessionFileFilter;

@SuppressWarnings("serial")
public class SelectSessionFilesDialog extends JDialog implements ActionListener {

	private JButton mBrowseButton = new JButton("Browse");
	private JButton mAddButton = new JButton("Add");
	private JButton mDeleteButton = new JButton("Delete");
	private JButton mOkButton = new JButton("Ok");
	private JButton mCancelButton = new JButton("Cancel");
	
	private JTextField mFilenameField = new JTextField(15);
	
	private DefaultListModel mListModel = new DefaultListModel();
	private JList mFileList = new JList(mListModel);
	private String mFileDirectory = ".";
	
	private FileFilter fileFilter;
	private Vector<String[]> sessionProfiles;
	private JTable table;
	private DefaultTableModel model;
	
	private boolean addingNewGroup = false;
	private int group = -1;
	
	private JLabel mTitle;
	
	private JFrame frame;
	
	// A new dialog used by the AnalysisFrame and NpairsAnalysisFrame classes
	// for adding session files.
	public SelectSessionFilesDialog(JFrame frame, FileFilter fileFilter, Vector<String[]> sessionProfiles, JTable table, DefaultTableModel model) {
		super(frame, "Select Session Files");
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		this.fileFilter = fileFilter;
		this.sessionProfiles = sessionProfiles;
		this.table = table;
		this.model = model;
		this.frame = frame;
		
		mTitle = new JLabel("Session Files: 0 selected");
		
		add(mTitle);
		
		JScrollPane filesScrollPane = new JScrollPane(mFileList);

		add(filesScrollPane);
		
		JPanel newFilePanel = new JPanel();
		newFilePanel.setLayout(new BoxLayout(newFilePanel, BoxLayout.X_AXIS));
		newFilePanel.add(mFilenameField);
		newFilePanel.add(mBrowseButton);

		add(newFilePanel);
		
		JPanel addDeletePanel = new JPanel();
		addDeletePanel.setLayout(new BoxLayout(addDeletePanel, BoxLayout.X_AXIS));
		addDeletePanel.add(mAddButton);
		addDeletePanel.add(mDeleteButton);
		mDeleteButton.setEnabled(false);
		
		add(addDeletePanel);
		
		JPanel okCancelPanel = new JPanel();
		okCancelPanel.setLayout(new BoxLayout(okCancelPanel, BoxLayout.X_AXIS));
		okCancelPanel.add(mOkButton);
		okCancelPanel.add(mCancelButton);
		
		add(okCancelPanel);
		
		mBrowseButton.addActionListener(this);
		mAddButton.addActionListener(this);
		mDeleteButton.addActionListener(this);
		mOkButton.addActionListener(this);
		mCancelButton.addActionListener(this);
		
		filesScrollPane.setPreferredSize(new Dimension(620, 460 - okCancelPanel.getPreferredSize().height));
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mBrowseButton) {
			JFileChooser chooser = new JFileChooser(mFileDirectory);
			chooser.setFileFilter(fileFilter);
			chooser.setMultiSelectionEnabled(true);

			int option = chooser.showOpenDialog(this);
			if (option == JFileChooser.APPROVE_OPTION) {
				File[] files = chooser.getSelectedFiles();
				mFileDirectory = files[0].getParent();

				String fileList = "";
				for (int i = 0; i < files.length; i++) {
					fileList += files[i].getAbsolutePath() + "; ";
				}
				fileList = fileList.substring(0, fileList.length() - 2);

				mFilenameField.setText(fileList);

				if (frame instanceof NpairsAnalysisFrame) {
					// NPAIRS Condition (class) info will need to be updated by 
					// user in Deselect menu
					((NpairsAnalysisFrame)frame).setClassSelectUpdated(false);
					// No. input scans will have to be updated in analysis setup gui
					((NpairsAnalysisFrame)frame).updateNumScans();
				}
			}
		} else if (e.getSource() == mAddButton) {
			String filenames = mFilenameField.getText().trim();

			if (filenames.equals("")) {
				JOptionPane.showMessageDialog(null, "No session files are provided.");
				return;
			}

			String[] files = filenames.split(";");
			for (int i = 0; i != files.length; i++) {
				String filename = files[i].trim();

				File file = new File(filename);
				if (file.exists()) {
					if (!mListModel.contains(filename)) {
						mListModel.addElement(filename);
					}
					mDeleteButton.setEnabled(true);
					mFilenameField.setText("");
					
					mTitle.setText("Session Files: " + mListModel.getSize() + " selected");
					
				} else {
					JOptionPane.showMessageDialog(null, "Session file " + filename + " can not be found.");
				}
			}

			mFileList.repaint();

		} else if (e.getSource() == mDeleteButton) {
			Object[] objectsToRemove = mFileList.getSelectedValues();

			for (Object o : objectsToRemove) {
				mListModel.removeElement(o);
			}

			if (mListModel.getSize() < 1) {
				mDeleteButton.setEnabled(false);
			}
			
			mTitle.setText("Session Files: " + mListModel.getSize() + " selected");

			mFileList.repaint();

		} else if (e.getSource() == mOkButton) {
			int numItems = mListModel.getSize();

			// There should be at least one session file.
			if (numItems > 0 ) {
				String[] groupSessionProfiles = new String[numItems];
				String fileList;
				
				if (numItems == 1) {
					fileList = "1 file: ";
				} else {
					fileList = numItems + " files: ";
				}
				
				for (int i = 0; i < numItems; i++) {
					String filename = (String) mListModel.get(i);
					groupSessionProfiles[i] = filename;
					fileList += filename + "; ";
				}
				fileList = fileList.substring(0, fileList.length() - 2);

				
				if (addingNewGroup) {
					model.addRow(new Object[]{model.getRowCount() + 1, fileList});
					sessionProfiles.add(groupSessionProfiles);
					
					if(!updateDeselect()){
						model.removeRow(model.getRowCount() - 1);
						sessionProfiles.remove(groupSessionProfiles);
						return;
					}
					
					
					
				} else { //Update the group the user chose to edit.
					
					sessionProfiles.setElementAt(groupSessionProfiles, group);
					
					// Update Deselect menu options in NpairsAnalysisFrame
					if(!updateDeselect()){
//						sessionProfiles.remove(groupSessionProfiles);
						return;
					}
					
					table.setValueAt(fileList, group, 1);
				}

				setVisible(false);

				
			} else {
				JOptionPane.showMessageDialog(this, "The list of session files is empty. " +
				"Please include at least one session file.");
			}

		} else if (e.getSource() == mCancelButton) {
			setVisible(false);
		}
	}

	/**
	 * Update the selectable conditions for an npairs analysis.
	 * All session files across all groups must maintain the following three
	 * properties. 1) They must all have the same number of conditions and
	 * 2) they must all have the same conditions. 3) The ordering of the 
	 * conditions is important.
	 * 
	 */
	private boolean updateDeselect() {
		if (frame instanceof NpairsAnalysisFrame) {
			NpairsAnalysisFrame main = (NpairsAnalysisFrame)frame;
			boolean pass = main.menuBar.updateDeselectMenuConds(sessionProfiles);
			
			if(!pass){ 
				return false;
			}
			
			// Update split partition info in NpairsAnalysisFrame
			main.resamplingOptionsPanel.updateSplitPartitions();
//			// Update n vol info in NpairsAnalysisFrame
//			main.dataReducOptionsPanel.updateNVols(sessionProfiles);
		}
		return true;
	}

	// This method is invoked if the user decides to use the dialog to
	// add a brand new group of session files.
	public void addNewGroup() {
		mListModel.clear();
		mFilenameField.setText("");
		
		addingNewGroup = true;
		mDeleteButton.setEnabled(false);
		
		mTitle.setText("Session Files: 0 selected");
		
		setVisible(true);
	}
	
	/** This method is invoked if the user decides to use the dialog to
	 *  edit an existing group of session files.
	 *  @param group The index number of the group to edit.
	 */
	public void editGroup(int group) {
		mListModel.clear();
		mFilenameField.setText("");
		
		addingNewGroup = false;
		this.group = group;
		
		// Displays the current session files in the group first.
		String[] filenames = sessionProfiles.get(group);
		for (String filename : filenames) {
			mListModel.addElement(filename);
		}
		
		mDeleteButton.setEnabled(true);
		
		mTitle.setText("Session Files: " + filenames.length + " selected");
		
		setVisible(true);
	}
	
}
