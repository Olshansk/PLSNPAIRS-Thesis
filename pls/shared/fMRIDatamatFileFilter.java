package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class fMRIDatamatFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_fMRIdatamat.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "fMRI Datamat Files";
    }
}