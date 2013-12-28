package pls.chrome.result;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JOptionPane;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.shared.FilePathCheck;
import pls.shared.BfMRIDatamatFileFilter;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.fMRIDatamatFileFilter;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

public class PlsResultLoader extends ResultLoader {
	public PlsResultModel mPlsResultModel = null;

	/**
	 * Sets the session and datamat profiles to either the arguments provided
	 * if error is not true, or null if error is true. This is done because it
	 * has been decided that no plots will be loaded if ALL of their respective
	 * datamat and session files cannot be loaded as well. Setting the session
	 * and datamat arrays to null lets the plots know through ModelisApplicable
	 * not to load this respective file.
	 * @param error error indicator.
	 * @param sessionPro The sessionfiles to set.
	 * @param datamatPro The datamat files to set.
	 */
	private void setSessionOnError(boolean error,
								   ArrayList<ArrayList<String>> sessionPro,
								   ArrayList<ArrayList<String>> datamatPro){
			if (error) {
				mResultModel.setSessionProfiles(null);
				mPlsResultModel.setDatamatProfiles(null);
			} else {
				mResultModel.setSessionProfiles(sessionPro);
				mPlsResultModel.setDatamatProfiles(datamatPro);
			}
	}

	/**
	 * Display a warning to the user telling them about session/datamat files
	 * that failed to load.
	 * @param sessionNF the session files that failed to load.
	 * @param datamatNF the datamat files that failed to load.
	 */
	private boolean displayWarning(ArrayList<String> sessionNF,
			ArrayList<String> datamatNF) {

		/*String warning = "The Response Function Plot, Contrasts Information" +
				" Panel\nand the Temporal Brain Scores Plot will not be loaded "+
				"for\nthis file because ";*/
		String warning = "Some plots will not be loaded because ";
		boolean error = false;
		if (sessionNF.size() > 0) {
			error = true;
			warning += "the following session files could not be\nloaded:\n\n";
			for (String file : sessionNF) {
				warning += file + "\n";
			}
		}
		if (datamatNF.size() > 0) {
			error = true;
			if (warning.equals("Some plots will not be loaded because\n ")){
				warning  = "the following datamat files could not be loaded.\n"
				+"\n(Their corresponding session files were not loaded):\n\n";
			}else{
				warning += "\n The following datamat files could not be " +
					"loaded." +	"\n(Their corresponding session files " +
							"were not loaded):\n\n";
			}
			for (String file : datamatNF) {
				warning += file + "\n";
			}
		}

		if(error){
			JOptionPane.showMessageDialog(null,
					warning, "Error",
					JOptionPane.ERROR_MESSAGE);
		}
		return error;
	}

	/**
	 * Retrieves the parent directory from an absolute file ath.
	 * @param absolutePath
	 * @return the parent directory of the provided absolute path.
	 */
	public static String getPrefix(String absolutePath) {
		return new File(absolutePath).getParent();
	}

	public PlsResultLoader(String filename) {
		super(filename);
	}
	
	protected void addOtherRelevantFields() {		
	}

	protected void createResultModel() {
		mPlsResultModel = new PlsResultModel();
		mResultModel = mPlsResultModel;
	}



	/**
	 * Reads which session files belong to this result file. Session file
	 * information is parsed and the paths to these session files are saved.
	 */
	private void prepSessionFiles(){
		MLCell sp = (MLCell) mResultInfo.get("SessionProfiles");
		ArrayList<ArrayList<String>> sessionProfiles = new ArrayList<ArrayList<String>>();

		for (int i = 0; i < sp.getN(); i++) {
			ArrayList<String> currGroupProfiles = new ArrayList<String>();
			MLCell currSp = (MLCell) sp.get(i);
			for (int c = 0; c < currSp.getM(); c++) {
				String thisc = ((MLChar) currSp.get(c)).contentToString();
				int first = thisc.indexOf('\'') + 1;
				int last = thisc.lastIndexOf('\'');
				String profile = thisc.substring(first, last);
				
				/*profile = mResultModel.getFileDir() + File.separator 
							+ (new File(profile).getName());*/
				currGroupProfiles.add(profile);
			}
			sessionProfiles.add(currGroupProfiles);
		}
		mResultModel.setSessionProfiles(sessionProfiles);
	}

	/**
	 * Loads the session files and their respective datamat files.
	 *
	 * The first session file is sought for in the path contained in the
	 * mat file. If the file is not found there it is then sought 
	 * for in the current working directory and if it is not found there
	 * the user is prompted to search for the file. The resulting path to the
	 * session file is then used for all subsequent loaded session files. 
	 * When a file is not found at the last known working path the cwd is 
	 * checked and if it cannot be found there then the user is prompted to
	 * search for the file. When the file is found the path to that file 
	 * becomes the new session file path. 
	 * 
	 * The path to the corresponding datamat file is always the same path 
	 * used to find the session file.
	 * 
	 */
	private void loadSessionAndDataMat(){
		// Retrieves the suffix of the results file in order to determine
		// the suffix of the datamat file.
		String fileSuffix;
		if(mResultModel.getFilename().endsWith(BfMRIResultFileFilter.EXTENSION)) {
			fileSuffix = BfMRIDatamatFileFilter.EXTENSION;
		} else {
			fileSuffix = fMRIDatamatFileFilter.EXTENSION;
		}

		ArrayList<ArrayList<String>> subjectNames = mResultModel.getSubjectNames();
		ArrayList<ArrayList<String>> sessionProfiles = mResultModel.getSessionProfiles();
		ArrayList<ArrayList<String>> datamatProfiles = new ArrayList<ArrayList<String>>();
		MLStructure sessionInfo = null;
		FilePathCheck check = new FilePathCheck();

		String sessionPrefix = null;
		
		ArrayList<String> sessionNF = new ArrayList<String>(15);
		ArrayList<String> datamatNF = new ArrayList<String>(15);

		//locate the datamat file that corresponds to each session file.
		for (int i = 0; i != sessionProfiles.size(); i++) {
			ArrayList<String> currGroupProfiles  = sessionProfiles.get(i);
			ArrayList<String> currGroupDatamats = new ArrayList<String>();
			ArrayList<Integer> indicesToRemove = new ArrayList<Integer>();
			
			for (int j = 0; j != currGroupProfiles.size(); j++) {
				String sessionFile = currGroupProfiles.get(j);
				File sessionF = new File(sessionFile);

				//If we have never set a prefix, set it here.
				if(sessionPrefix == null){
					sessionPrefix = sessionF.getParent();
				}
				sessionFile = sessionPrefix + File.separator + 
							sessionF.getName();

				try {
					// Reads each session file in order to extract the prefix
					// of the datamat file name. Uses the directory of the
					// session file to create the datamat's file path as well.

					sessionFile = check.getExistingFilePath(check.SESSION,
							sessionFile, mResultModel.getFileDir());
					//Always set the prefix to the last found file and then
					//load the file if it was found, otherwise add it to the list
					//of files not found.
					if(sessionFile != null){
						sessionPrefix = getPrefix(sessionFile);

						sessionInfo = (MLStructure) new NewMatFileReader(sessionFile)
								.getContent().get("session_info");

						String datamatPrefix = ((MLChar) sessionInfo.getField
								("datamat_prefix")).getString(0);

						String datamatFilename = sessionPrefix + File.separator 
												+ datamatPrefix + fileSuffix;
						String dNotFound = datamatFilename;

						// Checks if the datamat file actually exists.
						if(!new File(datamatFilename).isFile()){
							datamatFilename = null;
						}

						if (datamatFilename != null) {
							currGroupDatamats.add(datamatFilename);
							//Update the session file storage with the new location
							//of the session file. This is the only place this
							//needs to be done because this is the only time
							//we aren't removing an index because we couldn't
							//find a datamat file.

							currGroupProfiles.set(j, sessionFile);

						} else {
							indicesToRemove.add(j);
							datamatNF.add(new File(dNotFound).getName());
						}
					}
					else{
						sessionNF.add(sessionF.getName());

						//If the session file does not exist, don't look for the
						//datamat file.
						indicesToRemove.add(j);
					}
				}
				catch(IOException e){
					JOptionPane.showMessageDialog(null,
						"An abnormal (IOException) condition occured " +
						"Information could not be read from file " +
						e.getMessage() + ".", "Error",
						JOptionPane.ERROR_MESSAGE);
					indicesToRemove.add(j);
				}
			}
			datamatProfiles.add(currGroupDatamats);

			// If a datamat file was unable to be found, then its
			// corresponding subject name is removed from the list
			// of subject names and its session profile is also removed
			// from the list of session profiles.
			ArrayList<String> currSubjectNames = subjectNames.get(i);
			for (int c = indicesToRemove.size() - 1; c >= 0; c--) {
				int index = indicesToRemove.get(c);
				
				currSubjectNames.remove(index);
				currGroupProfiles.remove(index);
				
			}
		}

		setSessionOnError(displayWarning(sessionNF,datamatNF),
				          sessionProfiles,
						  datamatProfiles);
	}

	private void createPsvPlot(){
		MLArray permResult = mResultInfo.get("perm_result");
		MLArray temp = null;
		// Create permuted singular values plot

		if (permResult.isStruct()) {
			MLStructure struct_perm_result = (MLStructure) permResult;

			double[][] sProb = null;
			temp = struct_perm_result.getField("s_prob");
			if (temp != null) {
				sProb = ((MLDouble)temp).getTransposeArray();
			} else {
				temp = struct_perm_result.getField("sprob");
				if (temp != null) {
					sProb = ((MLDouble)temp).getTransposeArray();
				}
			}
			mPlsResultModel.setSProbability(sProb);

			mPlsResultModel.setNumPermutations(((MLDouble) struct_perm_result.getField("num_perm")).get(0, 0).intValue());
		}
	}

	private void readBehavioralData(){
		//String contrastFile = mResultModel.getConstrastFilename();
		MLArray temp = null;

		// Reading the behavioral data.
		//if (contrastFile != null && contrastFile.equals("BEHAV")) {

			temp = mResultInfo.get("behavname");
			if (temp != null && temp.isCell() ) {
				ArrayList<ArrayList<String>> groupBehavNames = new ArrayList<ArrayList<String>>();
				MLCell struct_behavname = (MLCell) temp;

				ArrayList<String> behavNames = new ArrayList<String>();
				for (int j = 0; j != struct_behavname.getSize(); j++) {
					String subj_name = struct_behavname.get(j).contentToString();
					int first = subj_name.indexOf('\'') + 1;
					int last = subj_name.lastIndexOf('\'');
					behavNames.add(subj_name.substring(first, last));
				}

				int numGroups = mPlsResultModel.getNumSubjectList().length;
				for (int i = 0; i != numGroups; i++) {
					groupBehavNames.add(behavNames);
				}

				mPlsResultModel.setBehavNames(groupBehavNames);
			}else{
				return; //no behav data so just return.
			}

			mPlsResultModel.setBehavData(((MLDouble) mResultInfo.get("behavdata")).getArray());

		//}
	}

	private void loadBrainLV(){
		MLArray temp = null;

		temp = mResultInfo.get("brainlv");

		// Load brainLV data if possible
		if (temp != null) {
			mResultModel.addBrainData(BrainData.BRAINLV_STRING, ((MLDouble)temp).getArray() );
		}
	}

	private void loadBootStrap(){
		MLArray temp = null;
		// Load bootstrap data if possible
		MLArray boot_result = mResultInfo.get("boot_result");
		if (boot_result.isStruct()) {
			MLStructure struct_boot_result = (MLStructure) boot_result;

			double[][] origUsc = null;
			temp = struct_boot_result.getField("orig_usc");
			if (temp != null) {
				origUsc = ((MLDouble)temp).getArray();
			} else {
				temp = struct_boot_result.getField("origUsc");
				if (temp != null) {
					origUsc = ((MLDouble)temp).getArray();
				}
			}
			mPlsResultModel.setOrigUsc(origUsc);

			double[][] ulUsc = null;
			temp = struct_boot_result.getField("ulusc");
			if (temp != null) {
				ulUsc = ((MLDouble)temp).getArray();
			} else {
				temp = struct_boot_result.getField("ulUsc");
				if (temp != null) {
					ulUsc = ((MLDouble)temp).getArray();
				}
			}
			mPlsResultModel.setUlUsc(ulUsc);

			double[][] llUsc = null;
			temp = struct_boot_result.getField("llusc");
			if (temp != null) {
				llUsc = ((MLDouble)temp).getArray();
			} else {
				temp = struct_boot_result.getField("llUsc");
				if (temp != null) {
					llUsc = ((MLDouble)temp).getArray();
				}
			}
			mPlsResultModel.setLlUsc(llUsc);

			temp = struct_boot_result.getField("compare");
			if (temp != null) {
				double[][] bootstrap = null;
				bootstrap = ((MLDouble) temp).getArray();
				mResultModel.addBrainData(BrainData.BOOTSTRAP_STRING, bootstrap);
			}
		}
	}

	protected void loadOtherData() {
		prepSessionFiles();
		loadSessionAndDataMat();
		loadBrainLV();
		loadBootStrap();
		createPsvPlot();
		readBehavioralData();
	}
	
	public PlsResultModel getPlsResultModel() {
		return mPlsResultModel;
	}
}
