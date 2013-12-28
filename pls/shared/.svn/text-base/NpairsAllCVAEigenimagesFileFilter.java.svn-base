package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAllCVAEigenimagesFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.all.egimg.nii");
    }
    
    public String getDescription() {
        return "CVA eigenimages from full-data analysis";
    }
}
