package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsReproducibilityFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.summ.cc");
    }
    
    public String getDescription() {
        return "Npairs-J Reproducibility File (CC)";
    }
}
