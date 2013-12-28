package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAllChiSquaredFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.all.chi");
    }
    
    public String getDescription() {
        return "Canonical Variates from full-data analysis";
    }
}
