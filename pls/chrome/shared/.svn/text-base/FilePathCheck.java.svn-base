package pls.chrome.shared;

import java.awt.Component;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import pls.shared.AnalyzeImageFileFilter;
import pls.shared.BfMRIDatamatFileFilter;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.NiftiImageFileFilter;
import pls.shared.NpairsBlockSessionFileFilter;
import pls.shared.PetContrastFileFilter;
import pls.shared.PetDatamatFileFilter;
import pls.shared.PetSessionFileFilter;
import pls.shared.fMRIContrastFileFilter;
import pls.shared.fMRIDatamatFileFilter;
import pls.shared.fMRISessionFileFilter;

public class FilePathCheck {
	
	public static final String SESSION = "session";
	public static final String CONTRAST = "contrast";
	public static final String DATAMAT = "datamat";
	public static final String IMAGE = "image";
	
	// The boolean value indicating whether the user has selected
	// the "No To All" button whenever he/she has been prompted
	// to search for a missing file.
	private boolean noToAll = false;

	/**
	 * Verifies that the given file path is valid. If it is valid, then it is
	 * returned. Otherwise, searches for the file in the given current
	 * directory next, and if the file is there, returns the path to the file
	 * there instead. If the file is not there either, then the user is prompted
	 * to browse for it. If the user rejects, then null is returned.
	 * 
	 * @param fileType the type of file to be searched for, so the appropriate
	 * file filters will be given if browsing is required.
	 * @param newFilePath the current file path to be verified
	 * @param currentDirectory the directory used to search for the file if the
	 * given file path is incorrect
	 */
	public String getExistingFilePath(String fileType, String filePath, String currentDirectory) {
		
		String newFilePath = filePath;
		
		File file = new File(newFilePath);
		String fileName = file.getName();
		
		// Checks if the file exists at the given file path first.
		// If it does not exist, then searches for the file at the
		// stored current directory next.
		boolean fileFound = file.exists();
		if (!fileFound && currentDirectory != null) {
    		newFilePath = currentDirectory + File.separator + fileName;
			file = new File(newFilePath);
			fileFound = file.exists();
		}
		
		boolean runningPlsNotNpairs;
		if (fileType.equals(SESSION) || fileType.equals("datamat"))
			runningPlsNotNpairs = true;
		else runningPlsNotNpairs = false;
		
		// If the file still can not be found, then the user is
		// prompted to browse for it next. However, if the user has
		// been prompted before and he/she selected the "no to all"
		// option, then the user will not be prompted and it will be
		// assumed that the file can not be found at all.
		if (!fileFound && noToAll) {
			return null;
		} else if (!fileFound && runningPlsNotNpairs) {
		//} else if (!fileFound) {
			JOptionPane optionPane = 
					createPromptOptionPane("The " + fileType + " file "
					+ filePath + " could not be found."
					+ "\n\rWould you like to browse for it?");
			
			JDialog confirmDialog = optionPane.createDialog(null, "Error");
			confirmDialog.setAlwaysOnTop(true);
			confirmDialog.setVisible(true);
			
			int option;
			//user closed option pane without making a choice.
			if (optionPane.getValue() == null) {
				option = JOptionPane.NO_OPTION;
			} else {
				option = ((Integer) optionPane.getValue()).intValue();
			}
			
    		if (option == JOptionPane.YES_OPTION) {
    			JFileChooser chooser = new JFileChooser(currentDirectory);
				chooser.setSize(660, 480);

    			// Sets the proper file filter based on the type of file being
    			// searched (contrast/session/datamat).
    			if (fileType.equals(SESSION)) {
    				chooser.addChoosableFileFilter(new BfMRISessionFileFilter());
    				chooser.addChoosableFileFilter(new NpairsBlockSessionFileFilter());
    				chooser.addChoosableFileFilter(new PetSessionFileFilter());
					chooser.addChoosableFileFilter(new fMRISessionFileFilter());
    			} else if (fileType.equals(CONTRAST)){
    				chooser.addChoosableFileFilter(new fMRIContrastFileFilter());
    				chooser.addChoosableFileFilter(new PetContrastFileFilter());
    			} else if (fileType.equals(DATAMAT)){
    				chooser.addChoosableFileFilter(new PetDatamatFileFilter());
    				chooser.addChoosableFileFilter(new BfMRIDatamatFileFilter());
    				chooser.addChoosableFileFilter(new fMRIDatamatFileFilter());
    			} else if (fileType.equals(IMAGE)) {
    				chooser.addChoosableFileFilter(new NiftiImageFileFilter());
    				chooser.addChoosableFileFilter(new AnalyzeImageFileFilter());
    			}
    			
    			option = chooser.showDialog(null, "Select " + fileType + " file");
    			if (option == JFileChooser.APPROVE_OPTION) {
    				newFilePath = chooser.getSelectedFile().getAbsolutePath();
    			} else {
    				return null; //return null if file selection was canceled.
    			}
    		} else {
				//"no to all" was selected.
    			if (option == JOptionPane.CANCEL_OPTION) {
					noToAll = true;
				}
    			return null; //return if "no to all" or no was selected.
    		}
		}
		return newFilePath; //returns the new filepath.
	}
	
	public void resetNoOption() {
		noToAll = false;
	}
	
	public JOptionPane createPromptOptionPane(String message) {
		JOptionPane optionPane = new JOptionPane(message,
				   JOptionPane.WARNING_MESSAGE,
				   JOptionPane.YES_NO_CANCEL_OPTION);

		// Retrieves the cancel button and renames it to "No To All".
		Component[] components = optionPane.getComponents();
		for (int i = 0; i != components.length; i++) {
			if (components[i] instanceof JPanel) {
				JPanel panel = (JPanel) components[i];
				Component[] subComponents = panel.getComponents();
				for (int j = 0; j != subComponents.length; j++) {
					if (subComponents[j] instanceof JButton) {
						JButton button = (JButton) subComponents[j];
						if (button.getText().equals("Cancel")) {
							button.setText("No To All");
							button.setMnemonic('A');
							return optionPane;
						}
					}
				}
			}
		}

		return optionPane;
	}
	
}
