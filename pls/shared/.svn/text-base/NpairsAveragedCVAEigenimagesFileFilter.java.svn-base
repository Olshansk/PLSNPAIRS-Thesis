package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAveragedCVAEigenimagesFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.summ.avg.nii");
    }
    
    public String getDescription() {
        return "Standard CVA eigenimages averaged across all splits";
    }
}