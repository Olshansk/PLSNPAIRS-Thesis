package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class PlsAnalysisSetupFileFilter extends FileFilter {
	public static final String EXTENSION = "_PLSAnalysisSetup.mat";
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "PLS Analysis Setup Parameter Files";
    }
}
