package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAllCanonicalVariatesFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.all.can");
    }
    
    public String getDescription() {
        return "Canonical Variates from full-data analysis";
    }
}
