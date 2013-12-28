package pls.chrome.sessionprofile;

import pls.sessionprofile.RunInformation;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.border.Border;
import javax.swing.table.DefaultTableModel;

import nl.jj.swingx.gui.modal.JModalFrame;
import pls.chrome.shared.BaseSaveMenuBar;
import pls.shared.AnalyzeNiftiFileFilter;
import pls.shared.OnsetFileFilter;

@SuppressWarnings("serial")
public class EditRunsFrame extends JModalFrame implements ActionListener {
	
	private JTabbedPane tabs;
	private SessionProfileFrame sessionFrame;
	
	private JButton addRunButton;
	private JButton removeRunButton;
	private JButton saveRunButton;
	
	private JButton replicateButton;
	
	public EditRunsFrame(SessionProfileFrame sessionFrame) {
		super("Edit Runs");
		this.sessionFrame = sessionFrame;
		
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
	    
		tabs = new JTabbedPane(JTabbedPane.TOP, JTabbedPane.SCROLL_TAB_LAYOUT);
		
		if(sessionFrame.runInfo.size() == 0) {
			tabs.add(new EditRunsPanel(sessionFrame, null), "Run 1");
		} else {
			for(int i = 0; i < sessionFrame.runInfo.size(); i++) {
				tabs.add(new EditRunsPanel(sessionFrame, sessionFrame.runInfo.get(i)), "Run " + (i + 1));
			}
		}
	    
		addRunButton = new JButton("Add New Run", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		addRunButton.addActionListener(this);
	    
		removeRunButton = new JButton("Remove Current Run", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Delete16.gif")));
		removeRunButton.addActionListener(this);
	    
		saveRunButton = new JButton("Store And Continue", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/SendMail16.gif")));
		saveRunButton.addActionListener(this);
		
		replicateButton = new JButton("Replicate trial information across runs");
		replicateButton.addActionListener(this);
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(addRunButton);
		buttonPanel.add(removeRunButton);
		buttonPanel.add(saveRunButton);
		
		JPanel replicatePanel = new JPanel();
		replicatePanel.add(replicateButton);
		
		JPanel mainPanel = new JPanel(new BorderLayout());
		mainPanel.add(tabs, BorderLayout.NORTH);
		mainPanel.add(replicatePanel, BorderLayout.CENTER);
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
		add(mainPanel);
		
		setJMenuBar(new EditRunsMenuBar(this));
		
        pack();
        
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;

        setLocation(x, y);
        
        setResizable(false);
        setVisible(true);
        
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == addRunButton) {
			int runIndex = tabs.getTabCount();
			tabs.add(new EditRunsPanel(sessionFrame, null), "Run " + (runIndex + 1));
			tabs.setSelectedIndex(runIndex);
			
		} else if (e.getSource() == removeRunButton) {
			int runIndex = tabs.getSelectedIndex();
			if (runIndex < 0) {
				return;
			}
			tabs.remove(runIndex);
			for (int i = runIndex; i < tabs.getTabCount(); i++) {
				tabs.setTitleAt(i, "Run " + (i + 1));
			}
			
		} else if (e.getSource() == saveRunButton) {
			saveRunAction();
		} else if (e.getSource() == replicateButton) {
			replicateAction();
		}
	}
	
//	private void updateDatapaths(String parentDir){
//		sessionFrame.setLastPath(parentDir);
//		for (int i = 0; i < tabs.getTabCount(); i++){
//			EditRunsPanel c = ((EditRunsPanel) tabs.getComponentAt(i));
//			String imageDir;
//			imageDir = new File(c.dataDirectoryField.getText()).getName();
//			
//			c.dataDirectoryField.setText(parentDir + File.separator + imageDir);
//		}
//	}
	
	/**
	 * Action performed when the user hits the 'Replicate trial information
	 * across runs' button.
	 */
	private void replicateAction(){
		int runIndex = tabs.getSelectedIndex();
		if (runIndex < 0) { //no valid runs loaded so return.
			return;
		}
		
		if(tabs.getTabCount() > 1){
			int opt;
			opt = JOptionPane.showConfirmDialog(this, 
					"This run's onsets and lengths " +
					"will be replicated across ALL other " +
					"runs.\n Are you sure you " +
					"want to do this?", "Replicate run onsets/lengths?", 
					JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
			
			if(opt == JOptionPane.NO_OPTION) return;
		}
		
		EditRunsPanel current = (EditRunsPanel) tabs.getComponentAt(runIndex);
		for (int i = 0; i < tabs.getTabCount(); i++) {
			if (i == runIndex) {
				continue;
			}
			
			EditRunsPanel c = ((EditRunsPanel) tabs.getComponentAt(i));
			for (int j = 0; j < current.table.getRowCount(); j++) {
				String value = current.table.getValueAt(j, 1).toString().trim();
				c.table.setValueAt(value, j, 1);
			}
			
			if (sessionFrame != null && sessionFrame.isBlockedFmri) {
				for(int j = 0; j < current.table.getRowCount(); j++) {
					String value = current.table.getValueAt(j, 2).toString().trim();
					c.table.setValueAt(value, j, 2);
				}
			}
		}
	}
	
	/**
	 * Perform the action when the user hits the 'Store and Continue' button
	 */
	private void saveRunAction(){
		Vector<RunInformation> runInfo = new Vector<RunInformation>();
		
		//For each run tab.
		for (int i = 0; i < tabs.getTabCount(); i++) {
			EditRunsPanel c = ((EditRunsPanel) tabs.getComponentAt(i));
			
			if (c.table.getCellEditor() != null) {
				c.table.getCellEditor().stopCellEditing();
			}
			
			String dataDirectory = c.dataDirectoryField.getText().trim();
			
			if (dataDirectory.equals("")) {
				JOptionPane.showMessageDialog(null, 
						"Data directory must be selected.", 
						"Error", JOptionPane.ERROR_MESSAGE);
				return;
			
			}
			
			String dataFiles = c.dataFilesField.getText().trim();
			if (dataFiles.equals("")) {
				JOptionPane.showMessageDialog(null, 
						"Data files must be selected.", 
						"Error", JOptionPane.ERROR_MESSAGE);
				return;
			}
			
			ArrayList<String> onsets = new ArrayList<String>();
			for (int j = 0; j < c.table.getRowCount(); j++) {
				String condition = "Condition " + 
						c.table.getValueAt(j, 0).toString().trim() 
						+ " of Run " + (i + 1);
				String onsetsLine = c.table.getValueAt(j, 1).toString().trim();
				
				if (!areOnsetValuesValid(onsetsLine, condition)) {
					return;
				}
				onsets.add(onsetsLine);
			}
			
			RunInformation currRunInfo = new RunInformation(dataDirectory,
															dataFiles, 
															onsets);
			
			if (sessionFrame != null && sessionFrame.isBlockedFmri) {
				ArrayList<String> lengths = new ArrayList<String>();
				for(int j = 0; j < c.table.getRowCount(); j++) {
					String condition = "Condition " 
						+ c.table.getValueAt(j, 0).toString().trim() 
						+ " of Run " + (i + 1);
					
					String lengthsLine = c.table.getValueAt(j, 2)
														.toString().trim();
					
					if (!areLengthValuesValid(lengthsLine, condition)) {
						return;
					}
					
					String onsetsLine = onsets.get(j);
					if (!sameLength(onsetsLine, lengthsLine, condition)) {
						return;
					}
					
					lengths.add(lengthsLine);
				}
				currRunInfo.lengths = lengths;
			}
			
			runInfo.add(currRunInfo);
		}
		if (sessionFrame != null) {
			sessionFrame.updateRuns(runInfo);
		}
		dispose();
	}
	
	// Clears all the fields in the currently-selected run tab.
	private void clearFieldsForSelectedTab() {
		int runIndex = tabs.getSelectedIndex();
		if (runIndex < 0) {
			return;
		}
		
		EditRunsPanel current = (EditRunsPanel) tabs.getComponentAt(runIndex);
		current.numScansLabel.setText("Number of Files: ");
		current.dataDirectoryField.setText("");
		current.dataFilesField.setText("");
		
		for (int i = 0; i < current.table.getRowCount(); i++) {
			current.table.setValueAt("", i, 1);
		}
		
		if (sessionFrame != null && sessionFrame.isBlockedFmri) {
			for (int i = 0; i < current.table.getRowCount(); i++) {
				current.table.setValueAt("", i, 2);
			}
		}
	}
	
	private boolean sameLength(String onsetsLine, String lengthsLine, String origin) {
		int numOnsets = onsetsLine.split(" ").length;
		int numLengths = lengthsLine.split(" ").length;
		
		boolean result = (numOnsets == numLengths);
		
		if (!result) {
			JOptionPane.showMessageDialog(null, origin + " does not contain an equal number of onset and length values.", "Error", JOptionPane.ERROR_MESSAGE);
			return false;
		} else {
			return true;
		}
	}
	
	private boolean areOnsetValuesValid(String line, String origin) {
		int prev = Integer.MIN_VALUE;
		
		String[] values = line.split(" ");
		for (int i = 0; i != values.length; i++) {
			try {
				int current = Integer.parseInt(values[i]);
				if (current == prev) {
					JOptionPane.showMessageDialog(null, origin + " contains a duplicate onset value: " + current, "Error", JOptionPane.ERROR_MESSAGE);
					return false;
				} else if (current < prev) {
					JOptionPane.showMessageDialog(null, origin + " has the onset value " + current + " after the onset value " + prev + ".", "Error", JOptionPane.ERROR_MESSAGE);
					return false;
				} else if (current < -1) {
					JOptionPane.showMessageDialog(null, origin + " should not have any onset values less than -1.", "Error", JOptionPane.ERROR_MESSAGE);
					return false;
				}
				
				prev = current;
			} catch (NumberFormatException e) {
				JOptionPane.showMessageDialog(null, origin + " contains an invalid onset value: " + values[i], "Error", JOptionPane.ERROR_MESSAGE);
				return false;
			}
		}
		return true;
	}
	
	private boolean areLengthValuesValid(String line, String origin) {
		String[] values = line.split(" ");
		for (int i = 0; i != values.length; i++) {
			try {
				int current = Integer.parseInt(values[i]);
				if (current < -1) {
					JOptionPane.showMessageDialog(null, origin + " should not have any length values less than -1.", "Error", JOptionPane.ERROR_MESSAGE);
					return false;
				}
			} catch (NumberFormatException e) {
				JOptionPane.showMessageDialog(null, origin + " contains an invalid length value: " + values[i], "Error", JOptionPane.ERROR_MESSAGE);
				return false;
			}
		}
		return true;
	}

final class EditRunsPanel extends JPanel {
	
	public JLabel numScansLabel;
	
    public JTextField dataDirectoryField = new JTextField();
    
    public JTextField dataFilesField = new JTextField();
    
    public JTable table = new JTable();
    
    private DefaultTableModel model = new DefaultTableModel();
	
	public EditRunsPanel(SessionProfileFrame sessionFrame, RunInformation runInfo) {
		
		numScansLabel = new JLabel("Number of Files: 0");
	    
	    dataDirectoryField.setColumns(48);
	    dataDirectoryField.setEditable(false);
	    
	    JLabel dataDirectoryLabel = new JLabel("Data Directory: ");
	    dataFilesField.setColumns(12);
	    dataFilesField.setEditable(false);
		JButton selectDataFilesButton = new JButton("Select Data Files", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		selectDataFilesButton.setIconTextGap(15);
	    
	    if(runInfo != null) {
	    	dataDirectoryField.setText(runInfo.dataDirectory);
	    	dataFilesField.setText(runInfo.dataFiles);
	    	numScansLabel.setText("Number of Files: " + runInfo.dataFiles.split(" ").length);
	    }
	    
	    selectDataFilesButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				
				/* If this run already has its data directory field set
				 * make sure the 'select data files' browser points to
				 * that location. If it is not set, then point to the
				 * parent directory of the previous run's data directory
				 * if it is a valid location. If the previous run's data
				 * directory is blank then look at the run previous to that
				 * run. If it is not a valid location or there is no previous
				 * run use the current working directory.
				 */
				
				String cwd = ".";
				if(!dataDirectoryField.getText().equals("")) {
					cwd = dataDirectoryField.getText();
				}else{
					
					int previousRunIndex = tabs.getSelectedIndex() - 1;
					while(previousRunIndex >= 0){
						EditRunsPanel pRun;
						pRun = (EditRunsPanel) tabs.getComponentAt(previousRunIndex);
						String parentD = pRun.dataDirectoryField.getText();
						
						if(parentD != null){
							parentD.trim();
							if(!parentD.equals("")){
								File parentDir = new File(new File(parentD).getParent());
								if(parentDir.exists()){
									cwd = parentDir.getAbsolutePath();
								}
								/*At this point we should have a valid browsing path
								 * for our new run. If we don't, stop looking for one
								 * from previous runs however. It will just confuse 
								 * the user and its easier to just default to the cwd.
								 */
								break; 
							}
						}
						previousRunIndex--;
					}
				}
				
				JFileChooser chooser = new JFileChooser(cwd);
			    chooser.setFileFilter(new AnalyzeNiftiFileFilter());
				chooser.setMultiSelectionEnabled(true);
				int option = chooser.showDialog(EditRunsPanel.this, "Select data files");
				if(option == JFileChooser.APPROVE_OPTION) {
					File[] sf = chooser.getSelectedFiles();
					if(sf.length > 0) {
						//Set to absolute path to the image directory.
						dataDirectoryField.setText(sf[0].getParent());
						
						//Update parent directories of all runs to match this
						//run's parent directory.
//						updateDatapaths(
//								new File(sf[0].getParent()).getParent()
//								);
					}
					
					String fileList = "";
					for(File f : sf) {
						fileList += " " + f.getName();
					}
					dataFilesField.setText(fileList.trim());
					numScansLabel.setText("Number of Files: " + sf.length);
				}
			}
		});
	    table.setModel(model);
	    table.setPreferredScrollableViewportSize(new Dimension(table.getPreferredScrollableViewportSize().width, 160));

	    model.addColumn("Condition Name");
	    model.addColumn("Onsets");
	    
	    if(sessionFrame.isBlockedFmri) {
	    	model.addColumn("Lengths");
	    }
    	
	    for(int i = 0; i < sessionFrame.conditionInfo.size(); i++) {
	    	String onsets = "";
	    	try {
	    		onsets = runInfo.onsets.get(i);
	    	} catch(Exception ex) {
	    		// No onsets exist
	    		onsets = "0";
	    	}
	    	if(sessionFrame.isBlockedFmri) {
		    	String lengths = "";
		    	try {
		    		lengths = runInfo.lengths.get(i);
		    	} catch(Exception ex) {
		    		// No onsets exist
		    		lengths = "0";
		    	}
	    		model.addRow(new String[]{sessionFrame.conditionInfo.get(i)[0], onsets, lengths});
	    	} else {
	    		model.addRow(new String[]{sessionFrame.conditionInfo.get(i)[0], onsets});
	    	}
	    }

	    JPanel directoryPanel = new JPanel(new BorderLayout());
	    directoryPanel.add(dataDirectoryLabel, BorderLayout.WEST);
	    directoryPanel.add(dataDirectoryField, BorderLayout.EAST);
	    JPanel labelPanel = new JPanel(new GridLayout(0, 1, 10, 10));
	    labelPanel.add(numScansLabel);
	    labelPanel.add(directoryPanel);
	    JPanel dataPanel = new JPanel(new GridLayout(1, 0));
	    dataPanel.add(dataFilesField);
	    dataPanel.add(selectDataFilesButton);
	    JPanel tablePanel = new JPanel();
	    Border border = BorderFactory.createTitledBorder("Event Onsets (TR)");
	    tablePanel.setBorder(border);
	    tablePanel.add(new JScrollPane(table));
	    
        JPanel mainPanel = new JPanel(new BorderLayout(15, 15));
        mainPanel.add(labelPanel, BorderLayout.NORTH);
        mainPanel.add(dataPanel, BorderLayout.CENTER);
        mainPanel.add(tablePanel, BorderLayout.SOUTH);
        add(mainPanel);
	}
}

	private class EditRunsMenuBar extends BaseSaveMenuBar {
		
		public EditRunsMenuBar(JFrame frame) {
			super(frame);
			setFileFilter(new OnsetFileFilter(), ".txt");
			
			// Retrieve the file menu to modify its options.
			JMenu fileMenu = getMenu(0);
			
			// Remove the "Save" menu item since the "Save As" one is needed only here.
			fileMenu.remove(1);
			
			// Rename the load/save menu items such that they are specifically for onset files.
			JMenuItem load = (JMenuItem) fileMenu.getMenuComponent(0);
			load.setText("Load Onsets from a text file for this run");
			
			JMenuItem save = (JMenuItem) fileMenu.getMenuComponent(1);
			save.setText("Save Onsets to a text file for this run");
			
			// Add a new menu item for clearing the run values in the current run tab.
			JMenuItem clear = new JMenuItem("Clear",
					new ImageIcon(this.getClass().getResource(
							"/toolbarButtonGraphics/general/Delete16.gif")));
			clear.setMnemonic('C');
			clear.getAccessibleContext().setAccessibleDescription("Clear");
			clear.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					clearFieldsForSelectedTab();
				}
			});
			fileMenu.add(clear, 2);
			
			fileMenu.insertSeparator(3);
		}
		
		public void load() {
			String name = new File(fileName).getName();
			try {
				int runIndex = tabs.getSelectedIndex();
				if (runIndex < 0) {
					return;
				}
				
				// Reads the onset values from the file first to verify
				// that they are all valid. This is to prevent setting
				// invalid values in the table.
				BufferedReader br = new BufferedReader(new FileReader(fileName));
				ArrayList<String> onsetValues = new ArrayList<String>();
				String line = br.readLine();
				while (line != null) {
					
					// Checks the line of onset values.
					if (!areOnsetValuesValid(line, "Onset file " + name)) {
						br.close();
						return;
					}
					String onsetsLine = line;
					onsetValues.add(line.trim());
					line = br.readLine();
					
					// Checks the line of length values next if the session file
					// is blocked fMRI.
					if (line != null && sessionFrame != null && sessionFrame.isBlockedFmri) {
						if (!areLengthValuesValid(line, "Onset file " + name)) {
							br.close();
							return;
						}
						String lengthsLine = line;
						
						// Checks if the number of onset values is the same as
						// the number of length values.
						if (!sameLength(onsetsLine, lengthsLine, "Onset file " + name)) {
							br.close();
							return;
						}
						
						onsetValues.add(line.trim());
						line = br.readLine();
					}
				}
				br.close();
				
				// Checks that the number of values in the file match the
				// required amount for the currently selected tab.
				EditRunsPanel current = (EditRunsPanel) tabs.getComponentAt(runIndex);
				if (sessionFrame != null && sessionFrame.isBlockedFmri) {
					if (onsetValues.size() < current.table.getRowCount() * 2) {
						JOptionPane.showMessageDialog(null, "Onset file " + name + " does not contain enough onset and length values for all the conditions.", "Error", JOptionPane.ERROR_MESSAGE);
				    	return;
					}
				} else {
					if (onsetValues.size() < current.table.getRowCount()) {
						JOptionPane.showMessageDialog(null, "Onset file " + name + " does not contain enough onset values for all the conditions.", "Error", JOptionPane.ERROR_MESSAGE);
				    	return;
					}
				}
				
				// Sets the table of the currently selected tab if there are
				// no errors.
				int k = 0;
				for (int i = 0; i < current.table.getRowCount(); i++) {
					line = onsetValues.get(k);
					current.table.setValueAt(line, i, 1);
					k++;
					
					if (sessionFrame != null && sessionFrame.isBlockedFmri) {
						line = onsetValues.get(k);
						current.table.setValueAt(line, i, 2);
						k++;
					}
				}
			} catch (Exception e) {
				JOptionPane.showMessageDialog(null, "Onset file " + name + " could not be loaded.", "Error", JOptionPane.ERROR_MESSAGE);
	    		return;
			}
		}
		
		public void save() {
			File file = new File(fileName);
			String name = file.getName();
			
			// Creates the selected file if it does not exist yet.
			if (!file.exists()) {
				try {
					file.createNewFile();
				} catch (Exception e) {
					JOptionPane.showMessageDialog(null, "Unable to create onset file " + name + ".", "Error", JOptionPane.ERROR_MESSAGE);
		    		return;
				}
			}
			
			try {
				int runIndex = tabs.getSelectedIndex();
				if (runIndex < 0) {
					return;
				}
				
				// Reads the onset values from the table of the
				// currently selected tab first to verify that they
				// are all valid.
				EditRunsPanel current = (EditRunsPanel) tabs.getComponentAt(runIndex);
				ArrayList<String> onsetValues = new ArrayList<String>();
				for (int i = 0; i < current.table.getRowCount(); i++) {
					String condition = "Condition " + current.table.getValueAt(i, 0).toString().trim();
					String onsetsLine = current.table.getValueAt(i, 1).toString().trim();
					
					// Checks the line of onset values.
					if (!areOnsetValuesValid(onsetsLine, condition)) {
						return;
					}
					onsetValues.add(onsetsLine);
					
					// Checks the line of length values next if the session file
					// is blocked fMRI.
					if (sessionFrame != null && sessionFrame.isBlockedFmri) {
						String lengthsLine = current.table.getValueAt(i, 2).toString().trim();
						if (!areLengthValuesValid(lengthsLine, condition)) {
							return;
						}
						
						// Checks if the number of onset values is the same as
						// the number of length values.
						if (!sameLength(onsetsLine, lengthsLine, condition)) {
							return;
						}
						
						onsetValues.add(lengthsLine);
					}
				}
				
				// Stores the values from the table of the currently selected
				// tab in the file if there are no errors.
				BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
				for (int i = 0; i != onsetValues.size(); i++) {
					String line = onsetValues.get(i);
					for (int j = 0; j != line.length(); j++) {
						bw.write(line.charAt(j));
					}
					bw.newLine();
				}
				bw.close();
			} catch (Exception e) {
				JOptionPane.showMessageDialog(null, "Onset file " + name + " could not be saved.", "Error", JOptionPane.ERROR_MESSAGE);
	    		return;
			}
		}
		
	}
	
}
