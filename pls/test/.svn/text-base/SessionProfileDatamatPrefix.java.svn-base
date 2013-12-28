package pls.test;

import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.chrome.shared.BaseSaveMenuBar;

/**
 * A class for testing that session files, when saved, always have the same
 * prefix as the datamat prefix. Session files should also always be saved
 * to the same directory as generated datamats unless the user only wants
 * to save the session file without generating datamats. Needless to say
 * however, when a user generates a datamat file, the user can expect to find
 * the complementary session file with the generated datamats with the same
 * prefix.
 *
 */
public class SessionProfileDatamatPrefix extends SessionProfileFrame{

	SessionProfileDatamatPrefix(){
		super(true, false, "Test frame");
	}

	/**
	 * 
	 * @param newPrefix the datamat prefix is set to this value.
	 */
	void setDatamatPrefix(String newPrefix){
		datamatPrefixField.setText(newPrefix);
	}
	
	void load(String fileToLoad){
		((BaseSaveMenuBar)getJMenuBar()).fileName = fileToLoad;
		((BaseSaveMenuBar)getJMenuBar()).load();
	}
	
	void save(){
		((BaseSaveMenuBar)getJMenuBar()).save();
	}
	
	void generateDatamats(){
		createStDatamatButton.getActionListeners()[0].actionPerformed(null);
		while(getDatamatGenerator().isAlive()){
			Thread.yield();
		}
	}
}
