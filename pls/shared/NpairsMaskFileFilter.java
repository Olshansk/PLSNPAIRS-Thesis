package pls.shared;

import java.io.File;
import java.io.FileFilter;

public class NpairsMaskFileFilter implements FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().toLowerCase().endsWith(".mask");
    }
    
    public String getDescription() {
        return "Mask files";
    }
}
