package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAllCVAEigenvaluesFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.all.egval");
    }
    
    public String getDescription() {
        return "CVA eigenvectors from full-data analysis";
    }
}
