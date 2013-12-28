package pls.othertools.niftiextractor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.fMRIResultFileFilter;


/**
 * Controller class for the nifti extractor program.
 * Part of the MVC design pattern.
 *
 */
class NiftiExtractorController {

	NiftiExtractorModel model;
	
	public NiftiExtractorController(){
		model = new NiftiExtractorModel();
	}
	
	/**
	 * Get Average Canonical data, Z-scored data, and Full reference data.
	 * @param resultFile the result file to retrieve this information from.
	 * @return the above 2d data at indexes 0,1,2 respectively.
	 */
	Map<String,double[][]> getNpairsData(String resultFile){
		return model.getNpairsData(resultFile);
	}
	
	/**
	 * Get bootstrap and brainlv data.
	 * @param resultFile the result file to retrieve this information from.
	 * @return 2d bootstrap data at index 0 and 2d brainlv data at index 1.
	 */
	Map<String,double[][]> getPlsData(String resultFile){
		return model.getPlsData(resultFile);
	}
	
	/**
	 * Get the number of lags for the specified result file.
	 * @param resultFile 
	 * @return The number of lags for the given result file. 
	 */
	int getWindowSize(String resultFile){
		return model.getWindowSize(resultFile);
	}
	
	/**
	 * Test whether the result file belonging to the given name is an npairs
	 * file.
	 * @param resultFile name of the loaded result file, matches the same name
	 * listed in the gui's file list.
	 * @return true if the specified result file is an npairs file, false 
	 * otherwise.
	 */
	boolean isNpairs(String resultFile){
		return model.isNpairs(resultFile);
	}
	
	/**
	 * Return the loaded resultfile file names.
	 * @return The loaded resultfile file names.
	 */
	Set<String> getResultFiles(){
		return model.getResultFiles();
	}
	
	/**
	 * Remove a result file
	 * @return True if the removal was successful
	 */
	boolean removeResultFile(String rf){
		return model.removeResultFile(rf);
	}
	
	/**
	 * 
	 * @param resultFile
	 * @return null if there was no error while extracting the images. 
	 * Return an error string otherwise.
	 */
	String extractData(File resultFile){
		try {
			model.addFile(resultFile);
		} catch (NiftiExtractorException e) {
			return e.getMessage();
		} catch (IOException e) {
			return e.getMessage();
		}
		return null;
	}
	
	String writeNiftiImage(String path, Set<String> pcFiles, 
						int cv, String datatype){
		try {
			model.writeImage(path, pcFiles, cv, datatype);
		} catch (IOException e) {
			/* Sometimes when the permissions are incorrect the error
			 * message will be 'No such file or directory'. I don't know
			 * why, and its not because it is a FileNotFoundException.
			 * At any rate it is a permissions issue so change the message. 
			 */
			String message = e.getMessage().replaceFirst(
					"No such file or directory",
					"Permission denied");
			return message;
		}
		return null;
	}
	
	String writeNiftiImage(File path, String resultFile, int lv, int[] lags, 
			String dataType) {

		if(!resultFile.contains(fMRIResultFileFilter.EXTENSION) &&
			!resultFile.contains(BfMRIResultFileFilter.EXTENSION) ){
			throw new IllegalArgumentException(
			"Chosen file name must end in an appropriate result file suffix");
		}

		try {
			String extractThis = resultFile;
			String newprefix;
			String suffix = null;
			String lagString="_lags";
			int i;
			
			i = resultFile.indexOf(fMRIResultFileFilter.EXTENSION);

			if (i != -1) {
				suffix = fMRIResultFileFilter.EXTENSION;
			}
			else {
				i = resultFile.indexOf(BfMRIResultFileFilter.EXTENSION);
				suffix = BfMRIResultFileFilter.EXTENSION;
			}

			newprefix = resultFile.substring(0, i);
			newprefix += "_" + dataType;
			newprefix += "_lv" + (lv+1);

			lagString += lagString(lags);
								
			newprefix += lagString + 
					suffix.substring(0,suffix.lastIndexOf(".mat"));
			resultFile = newprefix;
			
			resultFile += ".nii";

			path = new File(path.getAbsolutePath()
					+ File.separator + resultFile);
			

			model.writeImage(path,extractThis,lv,lags,dataType);
		} catch (IOException e) {
			/* Sometimes when the permissions are incorrect the error
			 * message will be 'No such file or directory'. I don't know
			 * why, and its not because it is a FileNotFoundException.
			 * At any rate it is a permissions issue so change the message. 
			 */
			String message = e.getMessage().replaceFirst(
					"No such file or directory",
					"Permission denied");
			return message;
		}
		return null;
	}
	
	/**
	 * Create a lag string out of a list of lags.
	 * @param lags
	 * @return
	 */
	String lagString(List<Integer> lags){
		int[] nums = new int[lags.size()];
		int i = 0;
		for(int n : lags){
			nums[i++] = n;
		}
		return lagString(nums);
	}
	
	/**
	 * Construct a lag string. I.e convert {4,5,6,8,10} into
	 * "4-6,8,10".
	 * @param lags The lags to create a string out of.
	 * @return The lag string.
	 */
	String lagString(int[] lags) {
		
		int laglen = lags.length;
		int preval = 0;
		boolean iscontinuous = false;
		StringBuilder retval = new StringBuilder(laglen*2+1);
		
		for (int i = 0; i < laglen; i++) {
			if (i == 0) {
				retval.append(lags[0]);
				preval = lags[0];
			} else {
				if (lags[i] == preval + 1) {
					if (i == laglen - 1) {
						retval.append("-");
						retval.append(lags[i]);
					} else {
						iscontinuous = true;
						preval = lags[i];
					}
				} else {
					if (iscontinuous) {
						retval.append("-");
						retval.append(preval);
					}
					retval.append("_");
					retval.append(lags[i]);
					preval = lags[i];
					iscontinuous = false;
				}
			}
		}
		return retval.toString();
	}
	
	/**
	 * Test if the passed in cv is valid.
	 * @param resultFile
	 * @param cv
	 * @param dataType
	 * @param isPLS
	 * @return
	 */
	boolean validCV(String resultFile, int cv, String dataType,boolean isPLS){
		if(isPLS)
			return cv <= getPlsData(resultFile).get(dataType)[0].length;
		return cv <= getNpairsData(resultFile).get(dataType)[0].length;
	}

	/**
	 * Empty the model of loaded data.
	 */
	void removeResultFiles(){
		model.clearData();
	}

}
