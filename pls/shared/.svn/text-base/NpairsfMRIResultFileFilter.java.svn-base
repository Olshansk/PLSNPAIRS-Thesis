package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsfMRIResultFileFilter extends FileFilter {
	
	public static final String EXTENSION = "_NPAIRSJresult.mat";
	
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(EXTENSION);
    }
    
    public String getDescription() {
        return "Npairs fMRI Result Files";
    }
}