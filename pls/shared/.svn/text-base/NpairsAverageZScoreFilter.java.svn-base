package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsAverageZScoreFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".cva.summ.zs-avg.nii");
    }
    
    public String getDescription() {
        return "CVA Z-scored eigenimages averaged across all splits";
    }
}
