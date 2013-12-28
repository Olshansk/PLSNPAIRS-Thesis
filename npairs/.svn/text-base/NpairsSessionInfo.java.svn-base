package npairs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import npairs.io.NpairsjIO;

import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.sessionprofile.RunInformation;
import pls.shared.MLFuncs;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;
import extern.niftijlib.Nifti1Dataset;

public class NpairsSessionInfo {
	
	/** a class containing session information to be stored in NpairsjSetupParams
	 * 
	 */
	
	private boolean debug = false;	
	
	private String sessFilename;

	/**
	 * Array containing filenames, including full path, of input data. Data can
	 * be in either 3D or 4D format. If the data is in 3D format, then the
	 * length of dataFileNames will equal the total number of scans in the data.
	 *  (TODO: Verify - If the data is
	 * in 4D format, then each 4D dataset must be for a single run. Hence the
	 * length of dataFileNames will equal the total number of runs in the data.)
	 * In both 3D and 4D cases, filenames must appear in ascending timepoint order.
	 */
	private Vector<String> vDataFilenames;
	
	private String datamatFilename;
	
//	private int[] skipTmpts; // 0-relative; tmpts are ordered from first scan of 
	                         // Run 1 to last scan of last Run in current session file
//	private int[] nTmptsPerFile;
	
	private String maskFilename;
	
	/** True if all voxels (>= 0) are to be included in masked data for current
	 *  session.
	 *  If no mask name is provided and inclAllVoxels = false 
	 *  then data is masked by using PLS-style thresholding technique.
	 */
	private boolean inclAllVoxels = false;
	
	/** 
	 *  Threshold to be used to determine data mask; if current sess
	 *  file has non-null mask filename, maskThreshVal is ignored;
	 *  if inclAllVoxels is true, maskThreshVal is set to 0.
	 */
	private double maskThreshVal;
	
	/**
	 * Number of 3D scans (timepoints) that are included in this session
	 * (excl skipped scans and runs and classes)
	 */
	private int numVols = 0;
	
	/** Number of 3D scans (timepoints) in session data 
	 * Num vols incl skipped scans and classes but excl skipped runs
	 */
	public int numTotalVols = 0;
	
	/** True if at least one of the files in dataFilenames has 4D data
	 */
	public boolean dataIs4D = false;
	
//	private boolean blocked = true;
	
	private int currNumRuns;

	private int numScansSkipped;

	private Vector<Integer> vInclRuns;

	private int inclRunCounter = 0;
	
	private Hashtable<String, Integer> classLabelMap;
	
	private MLStructure currSessFileInfo = null;
	
	private HashMap<Integer, Integer> classInclMap;

	private int nVolsExclSkipAndCls = 0;

	private int nVolsExclSkip = 0;
	
	private Vector<Integer> vVolLabels;

	private Vector<Integer> vClassLabels;
	
	private boolean useCondsAsClasses = true; // if false, load class info from file instead

	private Vector<Integer> vRunLabels;

	private int[] inclRuns;

	private int[] volLabels;

//	private int[] classLabels;

	private Vector<Integer> vSkipTmpts;

	private Vector<Integer> vNTmptsPerFile;

	private TreeMap<Integer, Integer> eventRelClassLabs;
	
	/** Class info is loaded into constructor 
	 * @param currSessFile matlab filename including path and suffix
	 * @param classLabelMap map of class names to integer labels
	 * @param condLabelsFromFile integer class labels - if overriding class labels in sess file
	 * @param conditionSelection array of integers length no. classes; 1 = include; 0 = exclude
	 * @param eventRelAnalysis boolean 
	 * @param inclRuns int array of runs to be included in current analysis (1-relative)
	 * @throws NpairsjException
	 */
	public NpairsSessionInfo(String currSessFile, Hashtable<String, Integer> classLabelMap,
			int[] condLabelsFromFile, int[] conditionSelection, boolean eventRelAnalysis, int[] inclRuns)
		throws NpairsjException {
		
		this.sessFilename = currSessFile;
//		this.blocked = isBlocked();
		this.classLabelMap = classLabelMap;
		if (condLabelsFromFile != null) {
			this.useCondsAsClasses = false;
		}
		
		vInclRuns = new Vector<Integer>();
		vVolLabels = new Vector<Integer>();
		vClassLabels = new Vector<Integer>();
		vRunLabels = new Vector<Integer>();
		vDataFilenames = new Vector<String>();
		vSkipTmpts = new Vector<Integer>();		
		vNTmptsPerFile = new Vector<Integer>();
		eventRelClassLabs = new TreeMap<Integer, Integer>();
		
		try {
			currSessFileInfo = (MLStructure)new NewMatFileReader(currSessFile).
				getContent().get("session_info");
		} 
		catch (Exception ex) {
			throw new NpairsjException("Session file " + currSessFile 
					+ " could not be loaded.");

		}

		// load datamat prefix
		String dMatSuffix = getDmatSuffix();
		String dMatPath = getSessFilePath(); // assume they're the same for now
		String currDmatPref = ((MLChar)currSessFileInfo.getField("datamat_prefix")).
			getString(0);
		datamatFilename = dMatPath + System.getProperty("file.separator")
			+ currDmatPref + "_" + dMatSuffix;

		if (debug) {
			System.out.println("Datamat filename: " + datamatFilename);
		}

		if (!eventRelAnalysis) {
			// first check if all voxels should be kept in curr mask
			try {
				inclAllVoxels = ((MLDouble)currSessFileInfo.getField("use_all_voxels")).get(0,0).
				intValue() == 1;
			}
			catch (NullPointerException npe) {
				// use_all_voxels not included yet in current sess file 
				inclAllVoxels = false;
			}
			if (!inclAllVoxels) {
				// load mask info for curr session (if available)
				String currMaskFile = "";
				try {
					currMaskFile = ((MLChar)currSessFileInfo.getField("mask")).
						getString(0);
					maskFilename = currMaskFile;
					maskThreshVal = -1;  // set to -1 when not used
				} 
				//				catch (NoSuchElementException ex) {
				//					// NPE thrown so NSEE handled in same way as NPE below
				//					throw new NullPointerException();
				//				}
				catch (NullPointerException ex) {
					// no mask supplied for curr session;
					// must be using thresholding instead
					maskFilename = null;
					try {
						double thr = ((MLDouble)currSessFileInfo.getField("brain_coord_thresh")).
						get(0,0).doubleValue();
						maskThreshVal = thr;
					}
					catch (NullPointerException ex2) {
						// brain_coord_thresh not included yet when this sess file was created
						maskThreshVal = -1;
					}
				}
			}
			else {
				maskThreshVal = -1;
			}
			//			
			// load run info for all runs in curr session
			MLStructure runStruct = ((MLStructure)currSessFileInfo.getField("run"));
			Vector<RunInformation> runInfo = SessionProfileFrame.
			getRunInformation(runStruct, true);  // true ==> isBlockedfMRI

			// determine excluded runs
			currNumRuns = ((MLDouble)currSessFileInfo.getField("num_runs")).
				get(0,0).intValue();
			String[] currRunsSkipped = ((MLChar)currSessFileInfo.getField("runs_skipped")).
			getString(0).split("\\s");
			
			// get incl runs info from session file if info not included in
			// NpairsSessionInfo constructor
			if (inclRuns == null) {
				for (int r = 1; r <= currNumRuns; ++r) {
					boolean skip = false;
					for (int s = 0; s < currRunsSkipped.length; ++s) {
						if (Integer.parseInt(currRunsSkipped[s]) == r) {
							skip = true;
						}
					}
					if (!skip) {
						vInclRuns.add(r);
					}
				}
				this.inclRuns = new int[vInclRuns.size()];
				for (int i = 0; i < vInclRuns.size(); ++i) {
					this.inclRuns[i] = vInclRuns.get(i);
				}
			}
			else {
				this.inclRuns = inclRuns;
			}
			
			// how many scans skipped?
			numScansSkipped = ((MLDouble)currSessFileInfo.getField("scans_skipped")).
			get(0,0).intValue();

			// get info for each included run
//			Iterator<Integer> iter = vInclRuns.iterator();
//
//			while (iter.hasNext()) {
//				int inclRun = iter.next();
			for (int inclRun : this.inclRuns) {
				++inclRunCounter;
				// get scan info for curr run
				String dataDir = runInfo.get(inclRun-1).dataDirectory;
				String[] dataFiles = runInfo.get(inclRun-1).dataFiles.split("\\s");
				int numDataFiles = dataFiles.length;

				int numScansInCurrRun = 0;
				for (int i = 0; i < numDataFiles; ++i) {
					String currFile = new File(dataDir, dataFiles[i]).
					getAbsolutePath();
					vDataFilenames.add(currFile);  // add 'em all and keep track of
					// which tmpts to skip
					if (debug && i == 0) {
						Npairsj.output.println("Curr file: " + currFile);
					}
					try {
						Nifti1Dataset n1ds = new Nifti1Dataset(currFile);
						n1ds.readHeader();
						int currNTmpts = n1ds.getTdim(); // keeps track of 3d/4d files
						if (currNTmpts == 0) currNTmpts = 1;
						vNTmptsPerFile.add(currNTmpts);
						numScansInCurrRun += currNTmpts;	
						if (currNTmpts > 1) {
							dataIs4D = true;
						}
//						if (debug) {
//							Npairsj.output.println("No. scans in " + dataFiles[i] + ": " +
//									currNTmpts);
//						}
					}
					catch (FileNotFoundException fnfe) {
						throw new NpairsjException("Image file " + currFile + " could not be found!" + 
						"\nCheck session file data paths.");
					}
					catch (IOException ioe) {
						throw new NpairsjException("Could not open image file "  + currFile + ".");
					}
				}

				if (debug) {
					Npairsj.output.println("Total no. scans in curr run: " + numScansInCurrRun);
				}

				// exclude skipped scans at beginning of run
				int currNumScansExclSkip = numScansInCurrRun - numScansSkipped;
				for(int i = 0; i < numScansSkipped; ++i) {
					vSkipTmpts.add(numTotalVols + i);
				}
				numTotalVols += numScansInCurrRun;

				// get class info for current run
				int[] currRunClassLabels = null;

//				if (classLabelMap == null) { // no class label map supplied; must create
//					                        // new one
//					classLabelMap = getClassLabelMap();
//				}
		
				classInclMap = getClassInclMap(classLabelMap, condLabelsFromFile, 
							conditionSelection);
				
				if (useCondsAsClasses) {
					currRunClassLabels = new int[currNumScansExclSkip];

					String[] condNames = MLFuncs.MLCell1dRow2StrArray
					((MLCell)currSessFileInfo.getField("condition"));
					ArrayList<String> condOnsets = runInfo.get(inclRun-1).onsets;
					ArrayList<String> condLengths = runInfo.get(inclRun-1).lengths;		

					for (int c = 0; c < condNames.length; ++c) {

						if (!classLabelMap.containsKey(condNames[c])) {
							throw new NpairsjException("Session condition labels do not match expected labels." +
							"\nAll session files in NPAIRS analysis must have same conditions.");
						}

						String[] currOnsets = condOnsets.get(c).split("\\s");
						String[] currLengths = condLengths.get(c).split("\\s");

						for (int i = 0; i < currOnsets.length; ++i) {
							if (Integer.parseInt(currOnsets[i]) >= 0) { // curr cond is included in this run
								for (int j = 0; j < Integer.parseInt(currLengths[i]); ++j) {
									int currIndex = Integer.parseInt(currOnsets[i]) + j
									- numScansSkipped;
									if (currIndex >= currNumScansExclSkip) {
										throw new NpairsjException("Onset labels out of "
												+ "scan range: " + currSessFile);
									}

									if (currIndex >= 0) {
										if (currRunClassLabels[currIndex] == 0) {
											// current scan has not yet been
											// labelled
											currRunClassLabels[currIndex] =  
												classLabelMap.get(condNames[c]).intValue();
										}
										else {
											throw new NpairsjException("Scan condition "
													+ "labels overlap - \n" + currSessFile);
										}
									}
								}
							}
						}
					}
					// Relabel remaining unlabelled scans with a '-1'  
					for (int i = 0; i < currRunClassLabels.length; ++i) {
						if (currRunClassLabels[i] == 0) {
							currRunClassLabels[i] = -1;
						}
					}
				}

				else {  // get class labels from input class file

					// TODO: classfile(s) should contain separate class label info for each run of each 
					// session file.
					// Currently assume all runs are of same length and that class labelling
					// provided in file applies to all runs.

					if (condLabelsFromFile.length != numScansInCurrRun) {
						throw new NpairsjException("Number of elements in class file" +
								" must be the same as the number of scans in each run " +
						"(including skipped scans at beginning).");
					}

					currRunClassLabels = new int[currNumScansExclSkip];
					for (int i = numScansSkipped ; i < numScansInCurrRun; ++i) {
						currRunClassLabels[i - numScansSkipped] = condLabelsFromFile[i];
					}
					
				}

					// append data info for current run
					if (debug) {
						Npairsj.output.println("Curr run class labels size: " + 
								currRunClassLabels.length);
					}

					for (int i = 0; i < currNumScansExclSkip; ++i) {
						++nVolsExclSkip;
						// check if curr scan condition is to be included
						int currClassLabel = currRunClassLabels[i];
						// note: if currClassLabel == -1, then current scan has
						// not been included in any conditions, hence should be
						// excluded from analysis
						if ((currClassLabel >= 0) && 
								(classInclMap.get(currClassLabel) == 1)) {
							++nVolsExclSkipAndCls;
							vVolLabels.add((inclRunCounter * numScansSkipped) + nVolsExclSkip - 1);

							//						vMaskFileNames.add(currMaskFile);

							//						vGrpLabels.add(grp);	
							//						vSessLabels.add(totalNumSess);
							vClassLabels.add(currClassLabel);
							vRunLabels.add(inclRunCounter);
						}
						else {
							// must exclude current scan's timepoint from data 
							vSkipTmpts.add(numTotalVols - currNumScansExclSkip + i);
						}
					}
				}
				
				
				
//			dataFilenames = new String[vDataFilenames.size()];
//			for (int i = 0; i < vDataFilenames.size(); ++i) {
//				dataFilenames[i] = vDataFilenames.get(i);
//			}
			
//			skipTmpts = new int[vSkipTmpts.size()];
//			for (int i = 0; i < vSkipTmpts.size(); ++i) {
//				skipTmpts[i] = vSkipTmpts.get(i);
//			}
			
			inclRuns = new int[vInclRuns.size()];
			for (int i = 0; i < vInclRuns.size(); ++i) {
				inclRuns[i] = vInclRuns.get(i);
			}
			
			volLabels = new int[vVolLabels.size()];
			for (int i = 0; i < vVolLabels.size(); ++i) {
				volLabels[i] = vVolLabels.get(i);
			}
			
//			classLabels = new int[vClassLabels.size()];
//			for (int i = 0; i < vClassLabels.size(); ++i) {
//				classLabels[i] = vClassLabels.get(i);
//			}
		
//			runLabels = new int[vRunLabels.size()];
//			for (int i = 0; i < vRunLabels.size(); ++i) {
//				runLabels[i] = vRunLabels.get(i);
//			}
			
//			nTmptsPerFile = new int[vNTmptsPerFile.size()];
//			for (int i = 0; i < vNTmptsPerFile.size(); ++i) {
//				nTmptsPerFile[i] = vNTmptsPerFile.get(i);
//			}
			
		}
		else {	// it's an event-related analysis: use datamats
			
			// get class info from current datamat 
			try {
			Map<String, MLArray> dMatInfo = new NewMatFileReader(datamatFilename, 
					new MatFileFilter(new String[]{"st_evt_list",
							"st_win_size", "create_datamat_info", })).getContent();
			int[] eventList = ((MLDouble)dMatInfo.get("st_evt_list")).getIntArray()[0];
			// NOTE length of eventList == no. rows in current [PLS-style] datamat;
			// multiply by winSize to get no. rows in current NPAIRS-style datamat
			int winSize = ((MLDouble)dMatInfo.get("st_win_size")).get(0, 0).intValue();
			// If data isn't merged across runs, then each run gets its own
			// event label in eventList. But we want each run to get the same
			// class labels.
			MLStructure createDmatInfo = (MLStructure)dMatInfo.get("create_datamat_info");
			boolean mergeRuns = ((MLDouble)createDmatInfo.getField("merge_across_runs")).
				getIntArray()[0][0] == 1;
			if (debug) {
				System.out.println("Merge runs? " + mergeRuns);
			}
			int[] runIdx = null;
			try {
				runIdx = ((MLDouble)createDmatInfo.getField("run_idx")).
					getIntFirstRowOfArray();				
			}
			catch (NullPointerException npe) {
				// "run_idx" variable is called "runIdx" in datamat files
				// generated in Java 
				runIdx = ((MLDouble)createDmatInfo.getField("runIdx")).
					getIntFirstRowOfArray();
			}
			inclRunCounter = runIdx.length;
			if (debug) {
				System.out.println("Num incl runs: " + inclRunCounter);
			}
			
			if (debug) {
				System.out.println("winSize: " + winSize);
			}
			
			int numEvents = eventList.length;
			int numConds = numEvents;
			if (!mergeRuns) {
				numConds = numEvents / inclRunCounter;
			    for (int c = 0; c < numConds; ++c) {
			    	for (int r = 0; r < inclRunCounter; ++r) {
			    		eventList[c*inclRunCounter + r] = c + 1;
			    	}
			    }
			}
			int clsLabLag0 = 1 - winSize;	// incremented to 1 at first iteration
			for (int eventLab : eventList) {
				if (debug) {
					System.out.println("Next event label: " + eventLab);
				}

				if (!eventRelClassLabs.containsKey(eventLab)) {
					clsLabLag0 += winSize;
					eventRelClassLabs.put(eventLab, clsLabLag0);
				}

				if (debug) {
					System.out.println("Including cond: " + eventLab);
					System.out.println("clsLabLag0 = " + clsLabLag0);
				}
				// each lag in a given condition (event) gets its own class
				for (int w = 0; w < winSize; ++w) {
					if (conditionSelection[eventRelClassLabs.size() - 1] == 1) {
						// curr condition is included
						if (debug) {
							System.out.println("Adding " + (clsLabLag0 + w) + "... ");
						}
						vClassLabels.add(clsLabLag0 + w);

						// Run is not an allowed split object for event-related
						// NPAIRS (right now) so runLabel info will not be used.
						// Just add '1' always to runLabels array for now.  
						vRunLabels.add(1); 
						// increment 'vols' info (actually no. rows in datamat) excluding
						// skipped classes (rows)
						++nVolsExclSkipAndCls;
					}
					else {
						// curr condition is excluded
						vSkipTmpts.add(numTotalVols);
					}
					++numTotalVols; // actually no. rows in concat. datamat before
					// excluding classes/conditions
				}

			}
			eventRelClassLabs.clear();
			
			nVolsExclSkip = nVolsExclSkipAndCls;
			
		} catch (IOException ioe) {
			throw new NpairsjException("Could not load datamat " + datamatFilename);
		}
		}
		numVols = nVolsExclSkipAndCls;
	}

	

	private boolean isBlocked() {
		if (sessFilename.endsWith("_BfMRIsession.mat")) {
			return true;
		}
		else {
			return false;
		}
	}

	private String getSessFilePath() {
		File f = new File(sessFilename);
		return f.getParent();
	}
	
	private String getDmatSuffix() {
		int begIdx = sessFilename.lastIndexOf("_");
		int endIdx = sessFilename.lastIndexOf("fMRI") + 4;
		String type = sessFilename.substring(begIdx + 1, endIdx);
		return type + "datamat.mat";
		
	}
	
	protected String getDatamatFilename() {
		return datamatFilename;
	}
	
	protected boolean inclAllVox() {
		return inclAllVoxels;
	}
	
	protected String getMaskFilename() {
		return maskFilename;
	}
	
	protected double getMaskThreshVal() {
		return maskThreshVal;
	}
	
	/** @return total number of data files entered into session file
	 * (including skipped scans, runs, classes)
	 */
	protected int getNDataFiles() {
		return vDataFilenames.size();
	}
	
	/** @return names of all data files entered into session file 
	 * (including skipped scans, runs, classes)
	 */
	protected Vector<String> getDataFilenames() {
		return vDataFilenames;
	}
	
//	/** Reads condition info from current sessFileInfo structure and returns map containing
//	 *  condition names as keys and (Integer) class labels as values.
//	 * @return classLabelMap 
//	 * @throws NpairsjException
//	 */
//	private Hashtable<String, Integer> getClassLabelMap() throws NpairsjException {
//	
//		String[] condNames = MLFuncs.MLCell1dRow2StrArray
//			((MLCell)currSessFileInfo.getField("condition"));
//
//		int nextClassLabel = 1;
//		Hashtable<String, Integer> classLabelMap = new Hashtable<String, Integer>(condNames.length);
//		for (int c = 0; c < condNames.length; ++c) {
//			if (!classLabelMap.containsKey(condNames[c])) {
//				classLabelMap.put(condNames[c], 
//						new Integer(nextClassLabel));
//				++nextClassLabel;
//			}
//		}
//		return classLabelMap;
//	}

	
	/** Returns HashMap indicating which condition (class) labels are to be included
	 *  in current analysis
	 * @param classLabelMap Hashtable containing class labels (Integers) as values corresponding
	 *                      to String class names as keys (null if cond labels loaded from file)
	 * @param condLabelsFromFile - array of integer class labels; null if cond labels not loaded from file
	 * @return HashMap with class labels as keys and 0 or 1 (indicating exclude/include, resp.) as values
	 */
	private HashMap<Integer, Integer> getClassInclMap(Hashtable<String, Integer> classLabelMap, 
			int[] condLabelsFromFile, int[] conditionSelection) {
		
		int numClasses = classLabelMap.size();
		
		int[] uniqClassLabs = null;
		if (numClasses == 0 && !(condLabelsFromFile == null)) {
			// class info is contained in condLabelsFromFile
			uniqClassLabs = MLFuncs.unique(condLabelsFromFile);
			numClasses = uniqClassLabs.length;
		}
		
		else {
			// class info is contained in classLabelMap
			uniqClassLabs = new int[numClasses];
			int k = 0;	
			for (Enumeration<Integer> e = classLabelMap.elements(); e.hasMoreElements() ; k++) {
				uniqClassLabs[k] = e.nextElement();
			}
		}
		
		uniqClassLabs = MLFuncs.sortAscending(uniqClassLabs);		
		HashMap<Integer, Integer> classInclMap = new HashMap<Integer, Integer>(numClasses);
		for (int i = 0; i < numClasses; ++i) {
			classInclMap.put(uniqClassLabs[i], conditionSelection[i]);
		}
		
		return classInclMap;
	}
	
	private void printParams() {
		System.out.println("Session filename: " + sessFilename);
		System.out.println("Length data filenames: " + vDataFilenames.size());
		System.out.println("Data filenames: ");
		NpairsjIO.print(vDataFilenames.toArray(new String[] {}));
	
		System.out.println("Datamat filename: " + datamatFilename);
		
		System.out.println("Length num tmpts per file array: " + vNTmptsPerFile.size());
		System.out.println("num tmpts per file: ");
		NpairsjIO.print(vNTmptsPerFile.toArray(new Integer[] {}));
	
		System.out.println("Mask filename: " + maskFilename);
		System.out.println("Include all voxels? " + inclAllVoxels);
	
		System.out.println("mask thresh val: " + maskThreshVal);
	
		System.out.println("Num vols excl skipped scans and runs and classes: " + numVols);
	
		System.out.println("Num vols incl skipped scans and classes but excl skipped runs: " + numTotalVols);
		
		System.out.println("Data is 4D? " + dataIs4D);
//		System.out.println("Blocked? " + !eventRelAnalysis);
		System.out.println("Number of runs: " + currNumRuns);
		System.out.println("Num scans skipped: " + numScansSkipped);

		System.out.println("Include which runs? ");
		NpairsjIO.print(inclRuns);
		
		System.out.println("Num incl runs: " + inclRunCounter);

		System.out.println("Class label map: ");
		System.out.println(classLabelMap.toString());
		
		System.out.println("Class inclusion map: ");
		System.out.println(classInclMap.toString());

		System.out.println("Num vols excluding skip and class: " + nVolsExclSkipAndCls);
		
		System.out.println("Volume labels: ");
		NpairsjIO.print(volLabels);
		System.out.println("Length of vol labels: " + volLabels.length);
		System.out.println("Skip tmpts: ");
		NpairsjIO.print(vSkipTmpts.toArray(new Integer[] {}));
		System.out.println("Length of skip tmpts: " + vSkipTmpts.size());
		boolean disjoint = true;
		for (Integer s : vSkipTmpts) {
			if (MLFuncs.contains(volLabels, s)) disjoint = false;
		}
		for (int v : volLabels) {
			if (vSkipTmpts.contains(v)) disjoint = false;
		}
		System.out.println("Skip tmpts and vol labels disjoint? " + disjoint);
		System.out.println("Length class labels array: " + vClassLabels.size());
		System.out.println("Class labels: ");
		NpairsjIO.print(vClassLabels.toArray(new Integer[] {}));
		System.out.println("Length run labels array: " + vRunLabels.size());
		System.out.println("Run labels: ");
		NpairsjIO.print(vRunLabels.toArray(new Integer[] {}));
	}
	
	public static void main(String[] args) {
		String sessFile = args[0];
		int[] conditionSelection = {1, 1, 0};
		Hashtable<String, Integer> classLabelMap = new Hashtable<String, Integer>();
		classLabelMap.put("ActR4", 1);
		classLabelMap.put("Baseline", 2);
		classLabelMap.put("Transition", 3);
		try {
			NpairsSessionInfo nsi = new NpairsSessionInfo(sessFile, classLabelMap, 
					null, conditionSelection, false, null);
			nsi.printParams();
			Vector<Integer> minNumVols = nsi.getNumRunVols();
			System.out.println("num vols per run: " + minNumVols.toString());
				
		} catch (NpairsjException e) {
			e.printStackTrace();
		}
	}


	/** @return no. of volumes included in each run entered into analysis;
	 * 	e.g. if runs 1, 2 and 4 from curr sess file are included in analysis and they include
	 * 65, 60 and 61 scans respectively, this method will return {65, 60, 61}. 
	 * These numbers reflect number of volumes actually included in analysis, 
	 * i.e. skipped scans and conditions/classes are excluded.
	 */
	public Vector<Integer> getNumRunVols() {
		Vector<Integer> runLabels = getRunLabels();
		Iterator<Integer> iter = runLabels.iterator();
		HashMap<Integer, Integer> numVolsPerRun = new HashMap<Integer, Integer>();
		// find number of vols included in each run
		while (iter.hasNext()) {
			int currLabel = (Integer) iter.next();
			if (!numVolsPerRun.containsKey(currLabel)) {
				numVolsPerRun.put(currLabel, 1);
			}
			else {
				// add one to curr label count
				numVolsPerRun.put(currLabel, numVolsPerRun.get(currLabel) + 1);
			}
		}
		int numRuns = numVolsPerRun.size();
		Vector<Integer> numVolsAsInteger = new Vector<Integer>(numRuns);
		numVolsAsInteger = new Vector<Integer>(numVolsPerRun.values());
//		int[] iNumVols = new int[numRuns];
//		for (int i = 0; i < numRuns; ++i) {
//			iNumVols[i] = numVolsAsInteger[i];
//		}
//		return iNumVols;
		return numVolsAsInteger;
	}



	/** @return total number of skipped timepoints in session file;
	 *  number of timepoints actually entered into analysis == 
	 *  number of skipped timepoints + getNumVols()
	 *  @see #getNumVols()
	 */
	public int getNSkipTmpts() {
		return vSkipTmpts.size();
	}

	/** @return indices (0-relative) of all skipped timepoints across
	 * all input data scans
	 */
	public Vector<Integer> getSkipTmpts() {
		return vSkipTmpts;
	}
	
	/** @return run labels for volumes included in analysis only (i.e.
	 *  excl skipped scans, runs and classes not in analysis)
	 */
	public Vector<Integer> getRunLabels() {
		return vRunLabels;
	}
	
	/** @return class labels for volumes included in analysis only (i.e.
	 *  excl skipped scans, runs and classes not in analysis)
	 */
	public Vector<Integer> getClassLabels() {
		return vClassLabels;
	}

	/** @return total number of timepoints per data file included
	 *  in session file (including skipped scans, runs, classes)
	 */
	public Vector<Integer> getNTmptsPerFile() {
		return vNTmptsPerFile;
	}

	public Boolean inclAllVoxels() {
		return inclAllVoxels;
	}

	/** @return Num vols excl skipped scans and runs and classes
	 *  (i.e. num vols actually entered into analysis)
	 *  @see #getNSkipTmpts()
	 */
	public int getNumVols() {
		return numVols;
	}

	/** @return Num vols incl skipped scans and classes but excl skipped runs
	 */
	public int getNumTotalVols() {
		return numTotalVols;
	}
	
	public int getNumInclRuns() {
		return inclRunCounter;
	}

}
