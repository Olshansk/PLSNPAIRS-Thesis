package pls.shared;

import javax.swing.JDialog;
import javax.swing.JOptionPane;

public class GlobalVariablesFunctions {
	public static final String PLS = "PLS";
	public static final String NPAIRS = "NPAIRS";
	public static String matrixLibrary = "ParallelColt";
	public static final String[] matrixLibraries = new String[]{"COLT", "ParallelColt", "ParallelColt (float)", "MATLAB"};
	private static final String plsnpairsVersion = "1.1.10(beta)";
	
	public static void showErrorMessage(String message, String title) {
		JOptionPane optionPane = new JOptionPane(message, JOptionPane.ERROR_MESSAGE);
		JDialog errorDialog = optionPane.createDialog(null, title);
		errorDialog.setAlwaysOnTop(true);
		errorDialog.setVisible(true);
	}
	
	public static void showErrorMessage(String message) {
		showErrorMessage(message, "Error");
	}
	
	public static String getVersion() {
		return plsnpairsVersion;
	}
}
