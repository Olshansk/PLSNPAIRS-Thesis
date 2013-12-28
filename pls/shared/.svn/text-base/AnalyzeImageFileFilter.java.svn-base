package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class AnalyzeImageFileFilter extends FileFilter {
	
	public static final String IMG_EXTENSION = ".img";
	public static final String HDR_EXTENSION = ".hdr";
	
    public boolean accept(File f) {
//        return f.isDirectory() || f.getName().endsWith(IMG_EXTENSION)
//        	|| f.getName().endsWith(HDR_EXTENSION);
		return f.isDirectory() || f.getName().endsWith(IMG_EXTENSION);
    }
    
    public String getDescription() {
        //return "Analyze Format Image Files";
		return ".img Format Image Files";
    }
}