package pls.sessionprofile;

import java.util.ArrayList;
import java.io.File;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import Jama.Matrix;

public class DatamatSaver {
	public DatamatSaver(Matrix stDatamat, int[] coords, int[] dims, double[] voxelSize, double[] origin, int[] stEventList, int windowSize, String sessionFile, boolean normalizeMeanVolume, boolean singleSubject, String brainMaskFile, double coordThresh, boolean considerAllVoxels, int numSkippedScans, int[] runIndeces, int[] ignoreSlices, boolean normalizeSignalMean, boolean mergeAcrossRunsFlag, String fileName) throws Exception {
		ArrayList<MLArray> list = new ArrayList<MLArray>();

	    if(normalizeMeanVolume) {
	        list.add(new MLDouble("normalize_volume_mean", new double[][]{{1}}));
	    } else {
	        list.add(new MLDouble("normalize_volume_mean", new double[][]{{0}}));
	    }
		list.add(new MLDouble("behavdata", new int[]{0,0}));
		list.add(new MLCell("behavname", new int[]{0,0}));
	    if(singleSubject) {
	        list.add(new MLDouble("SingleSubject", new double[][]{{1}}));
	    } else {
	        list.add(new MLDouble("SingleSubject", new double[][]{{0}}));
	    }
	    MLStructure createDatamatInfo = new MLStructure("create_datamat_info", new int[]{1, 1});
		createDatamatInfo.setField("brain_mask_file", new MLChar("brain_mask_file", brainMaskFile));
		createDatamatInfo.setField("brain_coord_thresh", new MLDouble("brain_coord_thresh", new double[][]{{coordThresh}}));
	    if(considerAllVoxels) {
	    	createDatamatInfo.setField("consider_all_voxels_as_brain", new MLDouble("consider_all_voxels_as_brain", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("consider_all_voxels_as_brain", new MLDouble("consider_all_voxels_as_brain", new double[][]{{0}}));
	    }
		createDatamatInfo.setField("num_skipped_scans", new MLDouble("num_skipped_scans", new double[][]{{numSkippedScans}}));
		//TODO: createDatamatInfo.setField("run_idx", new MLDouble("run_idx", new double[][]{{runIndex}}));
		if(runIndeces == null) {
			createDatamatInfo.setField("runIdx", new MLDouble("runIdx", new int[]{0,0}));
		} else {
			createDatamatInfo.setField("runIdx", new MLDouble("runIdx", MLFuncs.toDoubleArray(MLFuncs.plus(runIndeces, 1))));
		}
		if(ignoreSlices == null) {
			createDatamatInfo.setField("ignore_slices", new MLDouble("ignore_slices", new int[]{0,0}));
		} else {
			createDatamatInfo.setField("ignore_slices", new MLDouble("ignore_slices", MLFuncs.toDoubleArray(MLFuncs.plus(ignoreSlices, 1))));
		}
		createDatamatInfo.setField("temporal_window_size", new MLDouble("temporal_window_size", new double[][]{{windowSize}}));
	    if(normalizeMeanVolume) {
	    	createDatamatInfo.setField("normalize_volume_mean", new MLDouble("normalize_volume_mean", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("normalize_volume_mean", new MLDouble("normalize_volume_mean", new double[][]{{0}}));
	    }
	    if(normalizeSignalMean) {
	    	createDatamatInfo.setField("normalize_with_baseline", new MLDouble("normalize_with_baseline", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("normalize_with_baseline", new MLDouble("normalize_with_baseline", new double[][]{{0}}));
	    }
	    if(mergeAcrossRunsFlag) {
	    	createDatamatInfo.setField("merge_across_runs", new MLDouble("merge_across_runs", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("merge_across_runs", new MLDouble("merge_across_runs", new double[][]{{0}}));
	    }
	    if(singleSubject) {
	    	createDatamatInfo.setField("single_subject_analysis", new MLDouble("single_subject_analysis", new double[][]{{1}}));
	    } else {
	    	createDatamatInfo.setField("single_subject_analysis", new MLDouble("single_subject_analysis", new double[][]{{0}}));
	    }
	    list.add(createDatamatInfo);
	    list.add(new MLDouble("st_datamat", stDatamat.getArray()));
	    list.add(new MLDouble("st_evt_list", MLFuncs.toDoubleArray(MLFuncs.plus(stEventList, 1))));
	    list.add(new MLDouble("st_dims", MLFuncs.toDoubleArray(dims)));
	    list.add(new MLDouble("st_voxel_size", new double[][]{voxelSize}));
	    list.add(new MLDouble("st_origin", new double[][]{origin}));
	    list.add(new MLDouble("st_coords", MLFuncs.toDoubleArray(MLFuncs.plus(coords, 1))));
	    list.add(new MLDouble("st_win_size", new double[][]{{windowSize}}));
	    list.add(new MLChar("st_sessionFile", sessionFile));
        list.add(new MLChar("create_ver", "999999"));
	    
	    String path = new File(sessionFile).getParent();
	    String absoluteFileName = new File(path, fileName).getAbsolutePath();
	    
	    new MatFileWriter(absoluteFileName, list);
	}
}
