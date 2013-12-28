package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class fMRISessionFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_fMRIsession.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "Event-related fMRI Session Files";
    }
}