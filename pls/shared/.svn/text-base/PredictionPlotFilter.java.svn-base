package pls.shared;

import java.io.File;
import javax.swing.filechooser.FileFilter;

public class PredictionPlotFilter extends FileFilter{
	public static final String EXTENSION = ".txt";

	public boolean accept(File f) {
        return f.getName().endsWith(EXTENSION)  || f.isDirectory();
    }

	public String getDescription() {
        return "Prediction plot Median/Mean data.";
    }
}