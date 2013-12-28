package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class AnalyzeNiftiFileFilter extends FileFilter {
	public static final String IMG_EXTENSION = ".img";
	public static final String NIFTI_EXTENSION = ".nii";

	public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(IMG_EXTENSION) ||
				f.getName().endsWith(NIFTI_EXTENSION);
				
    }
    
    public String getDescription() {
        return "Analyze / Nifti Files";
    }
}

