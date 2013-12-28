package pls.shared;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class VoxelLocationFileFilter extends FileFilter {
    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(".txt");
    }
    
    public String getDescription() {
        return "Voxel Location Files";
    }
}
