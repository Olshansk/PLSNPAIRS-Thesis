package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class BfMRISessionFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_BfMRIsession.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "Blocked fMRI Session Files";
    }
}