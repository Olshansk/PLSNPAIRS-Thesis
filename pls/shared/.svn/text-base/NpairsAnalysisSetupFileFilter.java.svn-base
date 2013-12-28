package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsAnalysisSetupFileFilter extends FileFilter {
	public static String ext = "_NPAIRSAnalysisSetup.mat";
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(ext);
    }
    
    public String getDescription() {
        return "Npairs Analysis Setup Parameter Files";
    }
}
