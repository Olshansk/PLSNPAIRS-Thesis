package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class fMRIResultFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_fMRIresult.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "fMRI Result Files";
    }
}
