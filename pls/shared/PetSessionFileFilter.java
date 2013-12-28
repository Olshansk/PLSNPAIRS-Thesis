package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class PetSessionFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("_PETsession.mat");
    }
    
    public String getDescription() {
        return "PET Session Files";
    }
}