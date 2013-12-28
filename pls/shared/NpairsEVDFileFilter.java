package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsEVDFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith("EVD.evects");
        //	|| f.getName().endsWith("EVD.evals");
    }
    
    public String getDescription() {
        return "Npairs Initial EVD Files";
    }
}