package pls.othertools.niftiextractor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import npairs.io.NiftiIO;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.fMRIResultFileFilter;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

/**
 * This class performs the functions of both loading eigenimages from
 * result files and also saving the eigenimages to disk as a nifti file.
 * This class comprises the model of the MVC design pattern.
 *
 */
class NiftiExtractorModel {
	Map<String,ImageData> files = new TreeMap<String,ImageData>();
	
	/**
	 * Extracts the eigen images from a result file.
	 * Precondition: File is readable and is a result file.
	 * Postcondition: 1D volume information, dimensions, voxel size, and coords
	 * have been stripped from this file. 
	 * @param resultFile the result file to extract information from.
	 * @throws IOException 
	 * @throws NiftiExtractorException 
	 */
	void addFile(File resultFile) 
				throws NiftiExtractorException, IOException{
		
		//Exceptions thrown here.
		ImageData newData = extractImages(resultFile);
		
		files.put(resultFile.getName(),newData);
	}
	
	/**
	 * Writes the eigen images for the particular result file into a nifti file
	 * Precondition: resultFile is a valid string in the gui's combobox. Also
	 * resultFile is a key for imagedata that has been extracted previously,
	 * from a filename that is equal to resultFile.
	 * Path is a valid file path that the user has permissions to write to.
	 * @param path Filename to save to.
	 * @param resultFile
	 * @return
	 */
	void writeImage(File path, String resultFile, int lv, int[] lags,
			String dataType) 
			throws FileNotFoundException, IOException{
		
		PlsImageData fileData = (PlsImageData) files.get(resultFile);
		
		double[] lvdata = fetchData(fileData,lv,dataType);
		
		NiftiIO.writeVol4DPLS(lvdata, 
				fileData.st_coords, 
				true, 
				fileData.winSize,
				lags, 
				fileData.dims, 
				fileData.voxelSize,
				path.getAbsolutePath());
	
	}
	
	void writeImage(String path, Set<String> pcFiles, int cv, String dataType) 
									throws FileNotFoundException, IOException{
		
		NpairsImageData fileData = null;
		double[][] vol2d = new double[pcFiles.size()][];
		int i = 0;
		
		for(String rf : pcFiles){
			fileData = (NpairsImageData) files.get(rf);
			vol2d[i++] = fetchData(fileData,cv,dataType);
		}
		
		NiftiIO.writeVol4DNpairs(vol2d,
				fileData.st_coords,
				true,
				fileData.dims,
				fileData.voxelSize,
				path);
	}

	/**
	 * Returns the number of lags for the particular result file.
	 * @param resultFile
	 * @return The number of lags for the particular result file.
	 */
	int getWindowSize(String resultFile){
		return files.get(resultFile).winSize;
	}
	
	/**
	 * Get extracted eigen images for an npairs result file.
	 * @param resultFile
	 * @return The Average Canonical Eigen image, the z-scored image,
	 * and the Full reference image.
	 */
	Map<String,double[][]> getNpairsData(String resultFile){
		NpairsImageData nid = (NpairsImageData) files.get(resultFile);
		Map<String, double[][]> data = new HashMap<String, double[][]>(3);

		data.put(NiftiExtractorBatch.AVGCANON, nid.avg_canon);
		data.put(NiftiExtractorBatch.ZSCORE, nid.zscore);
		data.put(NiftiExtractorBatch.FULLREF, nid.full_ref);
		return data;
	}
	
	/**
	 * Get the extracted pls images.
	 * @param resultFile
	 * @return The bootstrap and brain lv images.
	 */
	Map<String,double[][]> getPlsData(String resultFile){
		PlsImageData pid = (PlsImageData) files.get(resultFile);
		Map<String, double[][]> data = new HashMap<String, double[][]>(3);

		data.put(NiftiExtractorBatch.BOOTSTRAP, pid.bootstrap);
		data.put(NiftiExtractorBatch.BRAINLV, pid.brainlv);
		return data;
	}
	
	/**
	 * Remove all loaded data from the model.
	 */
	void clearData(){
		Set<String> rfiles = new TreeSet<String>(files.keySet());
		for(String rfile : rfiles) files.remove(rfile);
	}
	
	/**
	 * Checks if the data stored under filename 'resultfile' is npairs or
	 * pls data.
	 * Precondition resultFile is a valid key to stored data.
	 * @param resultFile
	 * @return true if the data for this file is npairs data, false otherwise.
	 */
	boolean isNpairs(String resultFile){
		if(files.get(resultFile) instanceof NpairsImageData) 
			return true;
		return false;
	}
	
	/**
	 * Return the set of loaded result file names.
	 * @return The set of currently loaded result file names.
	 */
	Set<String> getResultFiles(){
		return files.keySet();
	}
	
	/**
	 * Remove a result file.
	 * @return True, given that the result file removed contained a mapping
	 * to non null data (the mapping should always be to non null data).
	 */
	boolean removeResultFile(String rf){
		if(files.remove(rf) == null) return false;
		return true;
	}
	
	private double[] fetchData(NpairsImageData id, int cv, String datatype){
		double[] cvdata = null;
		
		if(datatype.equals(NiftiExtractorBatch.AVGCANON)){
			cvdata = MLFuncs.getColumn(id.avg_canon,cv);
		}
		else if(datatype.equals(NiftiExtractorBatch.ZSCORE)){
			cvdata = MLFuncs.getColumn(id.zscore,cv);
		}
		else if(datatype.equals(NiftiExtractorBatch.FULLREF)){
			cvdata = MLFuncs.getColumn(id.full_ref, cv);
		}else{
			assert false : "Unknown datatype";
		}
		
		return cvdata;
	}
	
	/**
	 * Extracts eigen images for a set of cvs and for a particular data type.
	 * When extracting pls data, only a single lv is extracted.
	 * @param id The entire set of extracted data for a particular result file.
	 * @param cvs The set of cvs/lv to extract data for.
	 * @param datatype The specified datatype to extract.
	 */
	private double[] fetchData(PlsImageData id, int lv,String datatype){
		
		double[] lvdata = null;
		
		if(datatype.equals(NiftiExtractorBatch.BOOTSTRAP)){
			//pls data allows only a single lv be extracted at a time.
			//this lv is specified in cvs[0].
			lvdata = MLFuncs.getColumn(id.bootstrap, lv); 
		}
		else if(datatype.equals(NiftiExtractorBatch.BRAINLV)){
			lvdata = MLFuncs.getColumn(id.brainlv, lv);
		}
		else{
			assert false : "Bad datatype"; //invalid datatype. why?
		}
		return lvdata;
	}
	
	/**
	 * Extract image information from the specified result file.
	 * @param resultFile The file to extract information from.
	 * @return Extracted image data.
	 * @throws NiftiExtractorException if when extracting, a necessary field
	 * in the matlab file does not exist (i.e st_coords). 
	 * @throws IOException
	 */
	private ImageData extractImages(File resultFile) 
						throws NiftiExtractorException, IOException{
		
		String filename = resultFile.getName();
		String[] npairsFields = {"st_coords", "npairs_result","brainlv",
								"st_dims","st_voxel_size"};
		String[] plsFields = {"st_coords","brainlv","boot_result",
							"st_dims","st_voxel_size","st_win_size"};
		String[] inUseField;
		boolean usingPls;
		
		if(filename.endsWith(NpairsfMRIResultFileFilter.EXTENSION)){
			usingPls = false;
			inUseField = npairsFields;
		}
		else if(filename.endsWith(fMRIResultFileFilter.EXTENSION) ||
				filename.endsWith(BfMRIResultFileFilter.EXTENSION)){
			inUseField = plsFields;
			usingPls = true;
		}else{
			//I don't believe this should reasonably ever happen considering
			//the user is forced to use a valid filter.
			throw new NiftiExtractorException("Unknown filetype loaded." +
					" Aborting");
		}
		
		MLArray tempArray;
		Map<String, MLArray> mResultInfo;
		int[] st_coords;
		int[] dims;
		double[] voxSize;
		int winSize;
		
		//This is where the IOException is thrown.
		mResultInfo = new NewMatFileReader(resultFile.getAbsolutePath(), 
				new MatFileFilter(inUseField)).getContent();
		
		tempArray = mResultInfo.get("st_coords");
		if(tempArray != null){
			st_coords = ((MLDouble) tempArray).getIntFirstRowOfArray();
		}else{
			throw new NiftiExtractorException("Missing crucial field: " +
					"st_coords. Aborting");
		}

		tempArray = mResultInfo.get("st_dims");
		if(tempArray != null){
			dims = ((MLDouble) tempArray).getIntFirstRowOfArray();	
		}else{
			throw new NiftiExtractorException("Missing crucial field: " +
			"st_dims. Aborting");
		}
		
		tempArray = mResultInfo.get("st_voxel_size");
		if(tempArray != null){
			voxSize = ((MLDouble) tempArray).getFirstRowOfArray();
		}else{
			throw new NiftiExtractorException("Missing crucial field: " +
			"st_voxel_size. Aborting");
		}
					
		if(!usingPls){winSize = 1;}
		else{
			tempArray = mResultInfo.get("st_win_size");
			if(tempArray != null){
				winSize = (int) ((MLDouble) tempArray)
										.getFirstRowOfArray()[0];
			}else{
				throw new NiftiExtractorException("Missing crucial field: "
				+ "st_win_size. Aborting");
			}
		}
					
		if(!usingPls){
			return extractNpairs(mResultInfo, st_coords, 
					dims, voxSize, winSize);
		}
		return extractPls(mResultInfo, st_coords, dims, voxSize, winSize);
	}

	/**
	 * Extract pls specific fields from a pls result file.
	 * @param mResultInfo
	 * @param st_coords
	 * @param dims
	 * @param voxSize
	 * @param winSize
	 * @return
	 */
	private ImageData extractPls(Map<String, MLArray> mResultInfo,
			int[] st_coords, int[] dims, double[] voxSize, int winSize) {
		MLStructure tempStruct;
		MLArray tempArray;
		double[][] brainlv = null;
		double[][] bootstrap = null;
		
		tempArray = mResultInfo.get("brainlv");
		if(tempArray != null){
			brainlv = ((MLDouble) tempArray).getArray();
		}
		
		tempStruct = (MLStructure) mResultInfo.get("boot_result");
		if(tempStruct != null){
			tempArray = tempStruct.getField("compare");
			if(tempArray != null){
				bootstrap = ((MLDouble) tempArray).getArray();
			}
		}
		
		return new PlsImageData(dims, voxSize, st_coords, winSize,
				bootstrap, brainlv);
	}

	/**
	 * Extract npairs specific fields from an npairs file.
	 * @param mResultInfo
	 * @param st_coords
	 * @param dims
	 * @param voxSize
	 * @param winSize
	 * @return
	 * @throws NiftiExtractorException
	 */
	private ImageData extractNpairs(Map<String, MLArray> mResultInfo,
			int[] st_coords, int[] dims, double[] voxSize, int winSize)
			throws NiftiExtractorException {
		MLStructure tempStruct;
		MLArray tempArray;
		double[][] full_ref = null;
		double[][] zscored = null;
		double[][] avg_canon = null;
		
		tempArray = mResultInfo.get("brainlv");
		if(tempArray != null){
			full_ref = ((MLDouble) tempArray).getArray();
		}
		
		tempStruct = (MLStructure) mResultInfo.get("npairs_result");
		if(tempStruct == null){
			throw new NiftiExtractorException("Missing crucial " +
					"structure: npairs_result. Aborting");
		}
		
		tempArray = tempStruct.getField("zscored_brainlv_avg");
		if(tempArray != null){
			zscored = ((MLDouble)tempArray).getArray();
		}
		
		tempArray = tempStruct.getField("cv_brainlv_avg");
		if(tempArray != null){
			avg_canon = ((MLDouble) tempArray).getArray();
		}
		
		return new NpairsImageData(dims, voxSize, st_coords, winSize,
				zscored, avg_canon, full_ref);
	}
	
	/**
	 * Abstract class holding image data common to both npairs and pls files.
	 *
	 */
	private abstract class ImageData{
		int[] dims;
		double[] voxelSize;
		int[] st_coords;
		int winSize;
		
		public ImageData(int[] dimensions, 
						double[] voxSize,
						int[] coords,
						int windowSize){
			dims = new int[] {dimensions[0],dimensions[1],dimensions[3]};
			voxelSize = voxSize;
			st_coords = coords;
			winSize = windowSize;
		}
	}
	
	/**
	 * A class for holding extracted PLS image data.
	 *
	 */
	private class PlsImageData extends ImageData{

		double[][] bootstrap;
		double[][] brainlv;
		
		public PlsImageData(int[] dimensions, double[] voxSize,
					int[] coords, int windowSize,
					double[][] bootstrap, double[][] brainlv) {
			
			super(dimensions, voxSize, coords, windowSize);
			this.bootstrap = bootstrap;
			this.brainlv = brainlv;
		}
	}

	/**
	 * A class for holding extracted Npairs image data.
	 *
	 */
	private class NpairsImageData extends ImageData{
		double[][] zscore;
		double[][] avg_canon;
		double[][] full_ref;
		
		public NpairsImageData(int[] dimensions, double[] voxSize,
				int[] coords, int windowSize, 
				double[][] zscore, double[][] avg_canon, double[][] full_ref){
			
			super(dimensions, voxSize, coords, windowSize);
			
			this.zscore = zscore;
			this.avg_canon = avg_canon;
			this.full_ref = full_ref;
		}
		
	}

}
