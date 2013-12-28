package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class NpairsSplitsInfoFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(".vols");
    }
    
    public String getDescription() {
        return "Npairs Splits Volume Information";
    }
}