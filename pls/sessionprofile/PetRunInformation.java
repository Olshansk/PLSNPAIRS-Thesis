package pls.sessionprofile;

public class PetRunInformation {
	public String dataDirectory = null;
	public String dataFiles = null;
	public String[] onsets = null;
	public String[] lengths = null;
	
	public PetRunInformation(String dataDirectory, String dataFiles, String[] onsets) {
		this.dataDirectory = dataDirectory;
		this.dataFiles = dataFiles;
		this.onsets = onsets;
	}
}
