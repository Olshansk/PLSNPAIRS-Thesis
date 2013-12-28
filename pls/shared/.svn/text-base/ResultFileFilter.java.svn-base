package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class ResultFileFilter extends FileFilter {

	@Override
	public boolean accept(File file) {
		String filename = file.getName();
		return file.isDirectory() || filename.endsWith(fMRIResultFileFilter.EXTENSION) ||
		filename.endsWith(BfMRIResultFileFilter.EXTENSION) ||
		filename.endsWith(NpairsfMRIResultFileFilter.EXTENSION); //||
//		filename.endsWith(NiftiImageFileFilter.EXTENSION) ||
//		filename.endsWith(AnalyzeImageFileFilter.IMG_EXTENSION) ||
//		filename.endsWith(AnalyzeImageFileFilter.HDR_EXTENSION);
	}

	@Override
	public String getDescription() {
		return "All Result Files";
	}
	
}
