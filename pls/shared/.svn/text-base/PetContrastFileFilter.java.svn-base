package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class PetContrastFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("_PETcontrast.txt");
    }
    
    public String getDescription() {
        return "PET Contrast Files";
    }
}
