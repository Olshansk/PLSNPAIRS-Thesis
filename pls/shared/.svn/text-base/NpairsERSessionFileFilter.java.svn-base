package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsERSessionFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_NPAIRS_fMRIsession.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "NPAIRS Event-related Session Setup Files";
    }
}