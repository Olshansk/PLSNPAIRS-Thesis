package pls.shared;

import java.io.File;
import javax.swing.filechooser.FileFilter;

public class PLSResultFileFilter extends FileFilter{
	private static final String FMRI = "_fMRIresult.mat";
	private static final String BFMRI = "_BfMRIresult.mat";

    public boolean accept(File f) {
        return f.isDirectory() || f.getName().endsWith(FMRI)
				|| f.getName().endsWith(BFMRI);
    }

    public String getDescription() {
        return "PLS Result Files";
    }
}
