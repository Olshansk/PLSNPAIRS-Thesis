package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsAnalysisSetupERFileFilter extends NpairsAnalysisSetupFileFilter {
	@SuppressWarnings("hiding")
	public static String ext = "_NPAIRSAnalysisSetupER.mat";
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("_NPAIRSAnalysisSetupER.mat");
    }
    
    public String getDescription() {
        return "Npairs Analysis Setup Parameter Files (Event-related)";
    }
}
