package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class fMRIContrastFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("_fMRIcontrast.txt");
    }
    
    public String getDescription() {
        return "fMRI Contrast Files";
    }
}
