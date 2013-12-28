package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NiftiFileFilter extends FileFilter {
	public static final String NIFTI_EXTENSION = ".nii";
	
	@Override
	public boolean accept(File f){
		return f.isDirectory() || f.getName().endsWith(NIFTI_EXTENSION);
	}
	
	@Override
	public String getDescription(){
		return "Nifti Files";
	}
}
