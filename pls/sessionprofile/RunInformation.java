package pls.sessionprofile;

import java.util.ArrayList;

public class RunInformation {
    public String dataDirectory = null;
    public String dataFiles = null;
    public ArrayList<String> onsets = null;
    public ArrayList<String> lengths = null;
	
    public RunInformation(String dataDirectory, String dataFiles,
        ArrayList<String> onsets) {
        this.dataDirectory = dataDirectory;
        this.dataFiles = dataFiles;
        this.onsets = onsets;
    }
}
