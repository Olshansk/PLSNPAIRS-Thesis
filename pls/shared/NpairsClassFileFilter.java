package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsClassFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(".class");
    }
    
    public String getDescription() {
        return "Npairs CVA Class Label Information";
    }
}