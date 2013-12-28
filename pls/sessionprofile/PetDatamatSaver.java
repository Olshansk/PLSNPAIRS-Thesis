package pls.sessionprofile;

import java.util.ArrayList;
import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;

import pls.chrome.shared.BaseSaveMenuBar;
import pls.shared.MLFuncs;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

import Jama.Matrix;

public class PetDatamatSaver {
	public PetDatamatSaver(Matrix tdatamat, int[] coords, int[] dims, double[] voxelSize, double[] origin, String sessionFile, boolean normalizeMeanVolume, String brainMaskFile, double coordThresh, boolean considerAllVoxels, boolean normalizeSignalMean, String fileName) throws Exception {
		ArrayList<MLArray> list = new ArrayList<MLArray>();

				
	    list.add(new MLDouble("behavdata", new int[]{0,0}));
		list.add(new MLCell("behavname", new int[]{0,0}));
	    		 
	    MLStructure createDatamatInfo = new MLStructure("create_datamat_info", new int[]{1, 1});
		createDatamatInfo.setField("brain_mask_file", new MLChar("brain_mask_file", brainMaskFile));
		createDatamatInfo.setField("brain_coord_thresh", new MLDouble("brain_coord_thresh", new double[][]{{coordThresh}}));
	 
		if(normalizeMeanVolume) {
		    	createDatamatInfo.setField("normalize_volume_mean", new MLDouble("normalize_volume_mean", new double[][]{{1}}));
		} else {
		   	createDatamatInfo.setField("normalize_volume_mean", new MLDouble("normalize_volume_mean", new double[][]{{0}}));
		}
		if(considerAllVoxels) {
	    	createDatamatInfo.setField("consider_all_voxels_as_brain", new MLDouble("consider_all_voxels_as_brain", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("consider_all_voxels_as_brain", new MLDouble("consider_all_voxels_as_brain", new double[][]{{0}}));
	    }
	    
	    list.add(new MLDouble("coords", MLFuncs.toDoubleArray(MLFuncs.plus(coords, 1))));	    
	    list.add(createDatamatInfo);
	    list.add(new MLChar("create_ver", "999999"));	    
	    list.add(new MLDouble("datamat", tdatamat.getArray()));
	    list.add(new MLDouble("dims", MLFuncs.toDoubleArray(dims)));
	    list.add(new MLDouble("origin", new double[][]{origin}));
	    list.add(new MLChar("session_file", sessionFile));
	    MLStructure sessionInfo = null;
		
	    try {
			sessionInfo = (MLStructure)new NewMatFileReader(sessionFile).getContent().get("session_info");
	    } catch(Exception ex) {
    		JOptionPane.showMessageDialog(null, "Session file " + sessionFile + " could not be loaded.", "Error", JOptionPane.ERROR_MESSAGE);
    		return;
		}
		list.add(sessionInfo);
		
		list.add(new MLDouble("singleprecision", new double[][]{{1}}));
		list.add(new MLDouble("voxel_size", new double[][]{voxelSize}));
		
		String path = new File(sessionFile).getParent()+"\\";
		File datamatfile = new File(path, fileName);
		String absoluteFileName = new File(path, fileName).getAbsolutePath();
		
		if (datamatfile.exists()){
			//same file with the same name exist
			int res = JOptionPane.showConfirmDialog(null, "File " + fileName + " is already exist. Are you sure you want to overwrite it?.", "Confirm File Overwrite", JOptionPane.YES_NO_OPTION);
			if(res == JOptionPane.YES_OPTION){
				new MatFileWriter(absoluteFileName, list);
			}
			else
			{
				JFileChooser chooser = new JFileChooser(path);
				int option = chooser.showDialog(null, "Save As");
				if(option == JFileChooser.APPROVE_OPTION) {
					fileName = chooser.getSelectedFile().getAbsolutePath();
					new MatFileWriter(fileName, list);
				}
			}
		}
		else{
			new MatFileWriter(absoluteFileName, list);
		}
	}
}
