package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class PetDatamatFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("_PETdatamat.mat");
    }
    
    public String getDescription() {
        return "PET Datamat Files";
    }
}
