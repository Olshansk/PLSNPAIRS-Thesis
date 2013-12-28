package pls.analysis;

import java.util.ArrayList;
import java.util.Vector;
import java.util.Hashtable;
import java.util.Enumeration;

import javax.swing.JOptionPane;

import pls.shared.MLFuncs;
import pls.shared.NpairsfMRIResultFileFilter;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

import Jama.Matrix;
import npairs.Npairsj;
import npairs.io.NpairsjIO;
import npairs.NpairsjSetupParams;
import npairs.utils.CVA;

import java.io.*;

public class ResultSaver {

	private ArrayList<MLArray> list = new ArrayList<MLArray>();

	// Npairsj
	public ResultSaver(Npairsj npairsj, String npairsjSetupParamsFileName)
			throws FileNotFoundException, IOException, Exception {
		this(npairsj, npairsjSetupParamsFileName, null);
	}
	
	public ResultSaver(Npairsj npairsj, String npairsjSetupParamsFileName, 
			String resultsFilePrefix) throws FileNotFoundException, IOException, Exception {
		
		MLStructure npairsSetupParamStruct = null;
		try {
			npairsSetupParamStruct = (MLStructure)new NewMatFileReader(
					npairsjSetupParamsFileName).getContent().get("npairs_setup_info");
		} 
		catch (FileNotFoundException e) {
			throw new FileNotFoundException("Npairs setup parameter file " + 
					npairsjSetupParamsFileName + " could not be found.");
		} 
		catch (IOException e) {
			throw new IOException ("Npairs setup parameter file " + npairsjSetupParamsFileName 
					+ " could not be loaded.");
		}

		// get sessionFile info
		MLStructure sessionFileInfo = (MLStructure)npairsSetupParamStruct.
			getField("session_file_info");
		int numGroups = sessionFileInfo.getN();

		Vector<String[]> sessionFiles = new Vector<String[]>(numGroups);
		for (int grp = 0; grp < numGroups; ++grp) {

			//TODO: check if MLCell2StringArray bug means sess file info is incorrectly saved
			//      in Npairsj results .mat file
			String[] currSessFiles = MLFuncs.MLCell1dRow2StrArray((MLCell)(sessionFileInfo.
					getField("session_files", grp)));
			sessionFiles.add(currSessFiles);			
		}

		// get condition selection info
		int[] conditionSelection = null;
		if (npairsj.getSetupParams().useCondsAsClasses) {
			MLDouble conditionSelectionInfo = (MLDouble)npairsSetupParamStruct.
				getField("class_selection");
			int numConds = conditionSelectionInfo.getM();
			conditionSelection = new int[numConds];
			for (int i = 0; i < numConds; ++i) {
				conditionSelection[i] = conditionSelectionInfo.get(i).intValue();
			}
		}
		// else class info was read in from file so leave 
		// conditionSelection = null
		
		// Note that NpairsResultSaverPrep (extends ConcatenateDatamat) does not 
		// actually concatenate
		// any datamats, since this is not required in an Npairsj analysis; 
		// it only prepares the variables that would normally be 
		// stored in a PLS datamat file (e.g. st_coords) so that they can be 
		// stuck into the Npairsj results mat file using 
		// addStObjects(ConcatenateDatamat)
		NpairsResultSaverPrep npairsjResultsPrep = new NpairsResultSaverPrep(
				sessionFiles, conditionSelection, npairsj.getSetupParams(), 
				npairsj.getDataLoader());

		list.add(new MLChar("ContrastFile", "NONE"));
		list.add(new MLChar("create_ver", "999999"));

		addStObjects(npairsjResultsPrep);

		addPermutationResult(null);
		addBootstrapResult(null);
		addNpairsCVAResult(npairsj, npairsjResultsPrep.conditions);
		addLvEventList(npairsjResultsPrep.eventList);
		
		String resultsFileName = resultsFilePrefix;
		if (resultsFileName == null) {
			resultsFileName = ((MLChar)npairsSetupParamStruct.
				getField("results_filename")).getString(0);
		}
		if (!resultsFileName.endsWith(NpairsfMRIResultFileFilter.EXTENSION)) {
				resultsFileName = resultsFileName.concat(
						NpairsfMRIResultFileFilter.EXTENSION);
		}

		new MatFileWriter(resultsFileName, list);
	}

	
	// for testing NpairsResultSaver:
	public static void main1(String[] args) {
		String npairsSetupParamFileName = args[0];
		try {
			new ResultSaver(null, npairsSetupParamFileName);
		}
		catch (Exception e) {
			JOptionPane.showMessageDialog(null, "Error writing Npairs results to file.", 
					"Error", JOptionPane.ERROR_MESSAGE);
			e.printStackTrace();
			System.exit(1);
		}
	}

	// Deviation
	public ResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			String resultsFile, PermutationResult permResult, ComputeDeviationPls mainPls, 
			BootstrapResult bootResult) throws Exception {

		addStObjects(st);

		list.add(new MLChar("ContrastFile", "NONE"));

		list.add(new MLChar("create_ver", "999999"));
		
		//For mean-centered task PLS
		list.add(new MLDouble("method", new double[][]{{ 1 }}));

		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addDeviationPlsResult(mainPls);

		addLvEventList(st.eventList);

		new MatFileWriter(resultsFile, list);
	}

	// Deviation for PET
	public ResultSaver(int imagingType, ConcatenateDatamat st, int numBootstraps, int numPermutations, String resultsFile, PermutationResult permResult, ComputeBehaviorPls mainPls, BootstrapResult bootResult) throws Exception 
	{
		//addDeviationPlsResult(mainPls);
		addDeviationPETPlsResult(mainPls);
		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addPETObjects(st);

		list.add(new MLChar("create_ver", "999999"));

		System.out.println("resultfilename"+resultsFile);
		new MatFileWriter(resultsFile, list);
	}

	
	// Nonrotated task
	public ResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, String contrastFile, String resultsFile, PermutationResult permResult, ComputeNonrotatedTaskPls mainPls, BootstrapResult bootResult, Matrix design) throws Exception {
		addStObjects(st);

		list.add(new MLChar("ContrastFile", contrastFile));

		list.add(new MLChar("create_ver", "999999"));
		
		//For non-rotated task PLS
		list.add(new MLDouble("method", new double[][]{{ 2 }}));
							
		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addTaskPlsResult(mainPls);

		addLvEventList(st.eventList);

		addDesign(design);

		new MatFileWriter(resultsFile, list);
	}

	// Nonrotated task for PET
	public ResultSaver(int imagingType, ConcatenateDatamat st, int numBootstraps, int numPermutations, String contrastFile, String resultsFile, PermutationResult permResult, ComputeNonrotatedTaskPls mainPls, BootstrapResult bootResult, Matrix design) throws Exception {

		addTaskPlsResult(mainPls);

		addBootstrapResult(bootResult);

		addPermutationResult(permResult);

		addDesign(design);

		addPETObjects(st);

		list.add(new MLChar("create_ver", "999999"));

		new MatFileWriter(resultsFile, list);
	}

	// Multiblock
	public ResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, double confidenceLevel, String behaviorFile, String resultsFile, PermutationResult permResult, ComputeMultiblockPls mainPls, BootstrapResult bootResult, int[] bscan) throws Exception {
		addStObjects(st);

		list.add(new MLChar("ContrastFile", "MULTIBLOCK"));

		list.add(new MLChar("create_ver", "999999"));

		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addMultiblockPlsResult(mainPls);

		addLvEventList(st.eventList);

		list.add(new MLDouble("bscan", bscan));

		list.add(new MLDouble("ismultiblock", new double[][]{{ 1 }}));	

		new MatFileWriter(resultsFile, list);
	}
	
	//Multiblock for PET
	public ResultSaver(int imagingType, ConcatenateDatamat st, int numBootstraps, 
			int numPermutations, double confidenceLevel, String behaviorFile, 
			String resultsFile, PermutationResult permResult, 
			ComputeMultiblockPls mainPls, BootstrapResult bootResult, int[] bscan) throws Exception {

		addMultiblockPlsResult(mainPls);

		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addPETObjects(st);

		list.add(new MLDouble("bscan", bscan));

		list.add(new MLDouble("ismultiblock", new double[][]{{ 1 }}));	

		list.add(new MLChar("create_ver", "999999"));

		new MatFileWriter(resultsFile, list);
	}

	// Behavior
	public ResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			double confidenceLevel, String behaviorFile, String resultsFile, 
			PermutationResult permResult, ComputeBehaviorPls mainPls, 
			BootstrapResult bootResult) throws Exception {
		
		addStObjects(st);

		list.add(new MLChar("ContrastFile", "NONE"));

		list.add(new MLChar("create_ver", "999999"));

		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addBehaviorPlsResult(mainPls);

		addLvEventList(st.eventList);

		new MatFileWriter(resultsFile, list);
	}
	
	// Behavior for PET
	public ResultSaver(int imagingType, ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			double confidenceLevel, String behaviorFile, String resultsFile, PermutationResult permResult, 
			ComputeBehaviorPls mainPls, BootstrapResult bootResult) throws Exception {

		addBehaviorPlsResult(mainPls);

		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addPETObjects(st);

		list.add(new MLChar("create_ver", "999999"));

		new MatFileWriter(resultsFile, list);
	}
	
	//Non-rotated Behavioural for fMRI
	public ResultSaver(ConcatenateDatamat st, int numBootstraps,
			int numPermutations, double confidenceLevel,
			String behaviorFilename, String contrastFile, String resultsFile,
			PermutationResult permResult, ComputeNonRotatedBehavPls mainPls,
			BootstrapResult bootResult, Matrix design) throws Exception {
		
		list.add(new MLChar("ContrastFile", contrastFile)); //this has to be set to BEHAV for the behavioural data to load
		
		list.add(new MLChar("create_ver", "999999"));
		
		//For non-rotated behavioural PLS
		list.add(new MLDouble("method", new double[][]{{ 5 }}));
							
		addPermutationResult(permResult);

		addBootstrapResult(bootResult);

		addNonRotatedBehavPlsResult(mainPls);

		addDesign(design);
		
		addStObjects(st);

		addLvEventList(st.eventList);

		new MatFileWriter(resultsFile, list);
	}

	private void addStObjects(ConcatenateDatamat st) {
		// Save sessionProfiles
		MLCell sp = new MLCell("SessionProfiles", new int[]{1, st.sessionProfiles.size()});
		for(int i = 0; i < st.sessionProfiles.size(); i++) {
			MLCell currSp = new MLCell(null, new int[]{st.sessionProfiles.get(i).length, 1});
			for(int j = 0; j < st.sessionProfiles.get(i).length; j++) {
				currSp.set(new MLChar(null, st.sessionProfiles.get(i)[j]), j, 0);
			}
			sp.set(currSp, 0, i);
		}

		if(st.behavData != null) {
			// Save behavdata
			list.add(new MLDouble("behavdata", st.behavData.getArray()));

			// Save behavname
			MLCell behavname = new MLCell("behavname", new int[]{1, st.behavData.getColumnDimension()});
			for(int i = 0; i < st.behavData.getColumnDimension(); i++) {
				behavname.set(new MLChar("behavname" + i, "behav" + i + 1), 0, i);
			}
			list.add(behavname);

			// Save behavdata_lst
			MLCell behavdata_lst = new MLCell("behavdata_lst", new int[]{1, st.behavDataList.size()});
			for(int i = 0; i < st.behavDataList.size(); i++) {
				behavdata_lst.set(new MLDouble("behavdata_lst" + i, st.behavDataList.get(i).getArray()), 0, i);
			}
			list.add(behavdata_lst);
		}

		list.add(sp);

		list.add(new MLDouble("cond_selection", MLFuncs.toDoubleArray(st.conditionSelection)));
		list.add(new MLDouble("st_coords", MLFuncs.toDoubleArray(st.coords)));
		list.add(new MLDouble("st_dims", MLFuncs.toDoubleArray(st.dims)));
		list.add(new MLDouble("num_conditions", new double[][]{{st.numConditions}}));
		list.add(new MLDouble("st_win_size", new double[][]{{st.winSize}}));
		list.add(new MLDouble("st_voxel_size", MLFuncs.to2DArray(st.voxelSize)));
		list.add(new MLDouble("st_origin", MLFuncs.toDoubleArray(st.origin)));
		list.add(new MLDouble("subj_group", MLFuncs.toDoubleArray(st.subjectGroup)));

		// Save subjectName
		MLCell subj_name = new MLCell("subj_name", new int[]{1, st.subjectName.size()});
		for(int i = 0; i < st.subjectName.size(); i++) {
			subj_name.set(new MLChar("subj_name" + i, st.subjectName.get(i)), 0, i);
		}

		list.add(subj_name);

		// Save conditionNames
		MLCell cond_name = new MLCell("cond_name", new int[]{1, st.conditions.length});
		for(int i = 0; i < st.conditions.length; i++) {
			cond_name.set(new MLChar("cond_name" + i, st.conditions[i]), 0, i);
		}
		list.add(cond_name);

		list.add(new MLDouble("num_subj_lst", MLFuncs.toDoubleArray(st.numSubjectList)));
	}
	
	//for PET
	private void addPETObjects(ConcatenateDatamat st) {
		// Save sessionProfiles
		/*
        MLCell sp = new MLCell("SessionProfiles", new int[]{1, st.sessionProfiles.size()});
        for(int i = 0; i < st.sessionProfiles.size(); i++) {
    		MLCell currSp = new MLCell(null, new int[]{st.sessionProfiles.get(i).length, 1});
        	for(int j = 0; j < st.sessionProfiles.get(i).length; j++) {
            	currSp.set(new MLChar(null, st.sessionProfiles.get(i)[j]), j, 0);
        	}
        	sp.set(currSp, 0, i);
        }
		 */
		/*
		 * 
         if(st.behavData != null) {
        	// Save behavdata
        	list.add(new MLDouble("behavdata", st.behavData.getArray()));

            // Save behavname
            MLCell behavname = new MLCell("behavname", new int[]{1, st.behavData.getColumnDimension()});
            for(int i = 0; i < st.behavData.getColumnDimension(); i++) {
            	behavname.set(new MLChar("behavname" + i, "behav" + i + 1), 0, i);
            }
            list.add(behavname);

            // Save behavdata_lst
            MLCell behavdata_lst = new MLCell("behavdata_lst", new int[]{1, st.behavDataList.size()});
            for(int i = 0; i < st.behavDataList.size(); i++) {
            	behavdata_lst.set(new MLDouble("behavdata_lst" + i, st.behavDataList.get(i).getArray()), 0, i);
            }
            list.add(behavdata_lst);
        }

        list.add(sp);
        //list.add(new MLDouble("coords", MLFuncs.toDoubleArray(st.coords)));
        //list.add(new MLDouble("num_conditions", new double[][]{{st.numConditions}}));

		 */

		list.add(new MLDouble("newcoords", MLFuncs.toDoubleArray(st.coords)));
		list.add(new MLDouble("cond_selection", MLFuncs.toDoubleArray(st.conditionSelection)));
		list.add(new MLDouble("dims", MLFuncs.toDoubleArray(st.dims)));
		list.add(new MLDouble("voxel_size", MLFuncs.to2DArray(st.voxelSize)));
		list.add(new MLDouble("origin", MLFuncs.toDoubleArray(st.origin)));
		list.add(new MLDouble("num_cond_lst", MLFuncs.toDoubleArray(st.num_cond_lst)));
		list.add(new MLDouble("num_subj_lst", MLFuncs.toDoubleArray(st.numSubjectList)));        

		// Save subjectName
//		MLCell subj_name = new MLCell("subj_name_lst", new int[]{1, st.subjectName.length});
		MLCell subj_name = new MLCell("subj_name_lst", new int[]{1, st.numProfiles});


		for(int i = 0; i < st.numProfiles; i++) {
			subj_name.set(st.getSubjectName(i), i);
			//subj_name.set(new MLChar("subj_name_lst" + i,st.subjectName[i]), 0, i);
		}

		list.add(subj_name);

		// Save conditionNames
		/*
        MLCell cond_name = new MLCell("cond_name", new int[]{1, st.conditions.length});
        for(int i = 0; i < st.conditions.length; i++) {
        	cond_name.set(new MLChar("cond_name" + i, st.conditions[i]), 0, i);
        }
        list.add(cond_name);
		 */

		MLCell datamatFilesName = new MLCell("datamat_files", new int[]{1, st.numProfiles});
		MLCell datamatFilesTimestamp = new MLCell("datamat_files_timestamp", new int[]{1, st.numProfiles});

		System.out.println("Numprofiles" + st.numProfiles);
		for(int i = 0; i < st.numProfiles; i++) {
			datamatFilesName.set(new MLChar("datamat_files" + i, st.getDatamatFilename(i)), i);
			datamatFilesTimestamp.set(new MLChar("datamat_files_timestamp" + i, st.getDatamatFileTimeStamp(i)), i);
		}
		list.add(datamatFilesName);

		list.add(datamatFilesTimestamp);

	}
	
	
	private void addPermutationResult(PermutationResult permResult) {
		if(permResult != null) {
			MLStructure perm_result = new MLStructure("perm_result", new int[]{1, 1});

			perm_result.setField("num_perm", new MLDouble("num_perm", new double[][]{{permResult.numPermutations}}));

			if(permResult.permSample != null) {
				perm_result.setField("permsamp", new MLDouble("permsamp", MLFuncs.toDoubleArray(MLFuncs.plus(permResult.permSample, 1))));
			}
			if(permResult.sp != null) {
				perm_result.setField("sp", new MLDouble("sp", permResult.sp.getArray()));
			}
			if(permResult.sProb != null) {
				perm_result.setField("s_prob", new MLDouble("s_prob", permResult.sProb.getArray()));
			}
			if(permResult.sProb2 != null) {
				perm_result.setField("sprob", new MLDouble("sprob", permResult.sProb2.getArray()));
			}
			if(permResult.dp != null) {
				perm_result.setField("dp", new MLDouble("dp", permResult.dp.getArray()));
			}
			if(permResult.designLVprob != null) {
				perm_result.setField("designlv_prob", new MLDouble("designlv_prob", permResult.designLVprob.getArray()));
			}
			if(permResult.vProb != null) {
				perm_result.setField("vprob", new MLDouble("vPpob", permResult.vProb.getArray()));
			}
			if(permResult.TpermSamp != null) {
				perm_result.setField("Tpermsamp", new MLDouble("Tpermsamp", MLFuncs.toDoubleArray(MLFuncs.plus(permResult.TpermSamp, 1))));
			}
			if(permResult.BPermSamp != null) {
				perm_result.setField("Bpermsamp", new MLDouble("Bpermsamp", MLFuncs.toDoubleArray(MLFuncs.plus(permResult.BPermSamp, 1))));
			}
			if(permResult.posthocProb != null) {
				perm_result.setField("posthoc_prob", new MLDouble("posthoc_prob", permResult.posthocProb.getArray()));
			}
			list.add(perm_result);
		} else {
			list.add(new MLDouble("perm_result", new int[]{0,0}));
		}
	}

	
	private void addBootstrapResult(BootstrapResult bootResult) {
		if(bootResult != null) {
			MLStructure boot_result = new MLStructure("boot_result", new int[]{1, 1});
			
			boot_result.setField("num_boot", new MLDouble("num_boot", new double[][]{{bootResult.numBootstraps}}));

			if(bootResult.numLowVarBehavBoots != null) {
				boot_result.setField("num_LowVariability_behav_boots", new MLDouble("num_LowVariability_behav_boots", bootResult.numLowVarBehavBoots.getArray()));	
			}
			if(bootResult.bootSample != null) {
				boot_result.setField("bootsamp", new MLDouble("bootsamp", MLFuncs.toDoubleArray(MLFuncs.plus(bootResult.bootSample, 1))));
			}
			if(bootResult.brainStandardErrors != null) {
				boot_result.setField("brain_se", new MLDouble("brain_se", bootResult.brainStandardErrors.getArray()));
			}
			if(bootResult.brainLVStandardErrors != null) {
				boot_result.setField("brainlv_se", new MLDouble("brainlv_se", bootResult.brainLVStandardErrors.getArray()));
			}
			if(bootResult.designStandardErrors != null) {
				boot_result.setField("design_se", new MLDouble("design_se", bootResult.designStandardErrors.getArray()));
			}
			if(bootResult.compare != null) {
				boot_result.setField("compare", new MLDouble("compare", bootResult.compare.getArray()));
			}
			if(bootResult.compareDesign != null) {
				boot_result.setField("compare_design", new MLDouble("compare_design", bootResult.compareDesign.getArray()));	
			}
			if(bootResult.compareBehavLV != null) {
				boot_result.setField("compare_behavlv", new MLDouble("compare_design", bootResult.compareBehavLV.getArray()));	
			}
			if(bootResult.origCorr != null) {
				boot_result.setField("orig_corr", new MLDouble("orig_corr", bootResult.origCorr.getArray()));	
			}
			if(bootResult.origBrainLV != null) {
				boot_result.setField("orig_brainlv", new MLDouble("orig_brainlv", bootResult.origBrainLV.getArray()));	
			}
			if(bootResult.bScores2 != null) {
				boot_result.setField("b_scores2", new MLDouble("b_scores2", bootResult.bScores2.getArray()));	
			}
			if(bootResult.origUsc != null) {
				boot_result.setField("orig_usc", new MLDouble("orig_usc", bootResult.origUsc.getArray()));	
			}
			if(bootResult.ulUsc != null) {
				boot_result.setField("ulusc", new MLDouble("ulusc", bootResult.ulUsc.getArray()));	
			}
			if(bootResult.llUsc != null) {
				boot_result.setField("llusc", new MLDouble("llusc", bootResult.llUsc.getArray()));	
			}
			if(bootResult.ulUscAdj != null) {
				boot_result.setField("ulusc_adj", new MLDouble("ulusc_adj", bootResult.ulUscAdj.getArray()));	
			}
			if(bootResult.llUscAdj != null) {
				boot_result.setField("llusc_adj", new MLDouble("llusc_adj", bootResult.llUscAdj.getArray()));	
			}
			if(bootResult.ulCorr != null) {
				boot_result.setField("ulcorr", new MLDouble("ulcorr", bootResult.ulCorr.getArray()));	
			}
			if(bootResult.llCorr != null) {
				boot_result.setField("llcorr", new MLDouble("llcorr", bootResult.llCorr.getArray()));	
			}
			if(bootResult.ulCorrAdj != null) {
				boot_result.setField("ulcorr_adj", new MLDouble("ulcorr_adj", bootResult.ulCorrAdj.getArray()));	
			}
			if(bootResult.llCorrAdj != null) {
				boot_result.setField("llcorr_adj", new MLDouble("llcorr_adj", bootResult.llCorrAdj.getArray()));	
			}
			if(bootResult.prop != null) {
				boot_result.setField("prop", new MLDouble("prop", bootResult.prop.getArray()));	
			}
			if(bootResult.distrib != null) {
				double[] flattened = null;
				for(Matrix M : bootResult.distrib) {
					flattened = MLFuncs.append(flattened, MLFuncs.flattenVertically(M.getArray()));
				}
				MLDouble distrib = new MLDouble("distrib", new int[]{bootResult.distrib[0].getRowDimension(), bootResult.distrib[0].getColumnDimension(), bootResult.distrib.length});
				for(int i = 0; i < flattened.length ; i++) {
					distrib.set(new Double(flattened[i]), i);
				}
				boot_result.setField("distrib", distrib);
			}
			if(bootResult.badBeh != null) {
				MLCell badbeh = new MLCell("badbeh", new int[]{1, bootResult.badBeh.size()});
				for(int i = 0; i < bootResult.badBeh.size(); i++) {
					badbeh.set(new MLDouble("badbeh" + i, bootResult.badBeh.get(i).getArray()), 0, i);
				}
				boot_result.setField("badbeh", badbeh);	
			}
			if(bootResult.countNewTotal != 0) {
				boot_result.setField("countnewtotal", new MLDouble("countnewtotal", new double[][]{{ bootResult.countNewTotal }}));	
			}
			
			//added for PET
			if(bootResult.numLowVariabilityBehavBoots != null) {
				boot_result.setField("num_LowVariability_behav_boots", new MLDouble("num_LowVariability_behav_boots", MLFuncs.toDoubleArray(MLFuncs.plus(bootResult.numLowVariabilityBehavBoots, 1))));
			}

			//added for PET
			if(bootResult.zeroBrainStandarErrors != null) {
				boot_result.setField("zero_brain_se", new MLDouble("zero_brain_se", MLFuncs.toDoubleArray(MLFuncs.plus(bootResult.zeroBrainStandarErrors, 1))));
			}

			list.add(boot_result);
		} else {
			list.add(new MLDouble("boot_result", new int[]{0,0}));
		}
	}

	private void addDeviationPlsResult(ComputeDeviationPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("designlv", mainPls.designLV.getArray()));
		list.add(new MLDouble("b_scores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("d_scores", mainPls.designScores.getArray()));
	}

	private void addNpairsCVAResult(Npairsj npairsj, String[] condNames) throws IOException {
		
		// fill 'brainlv' with full-data cv eigenimage results
		list.add(new MLDouble("brainlv", npairsj.fullDataCVA.getEigimsBig().toArray()));
		
		// note that "s" is stored as double[k][1] (k == no. sing vals or eigenvals)
		// fill 's' with cv evals from full-data cv analysis
		list.add(new MLDouble("s", MLFuncs.transpose(npairsj.fullDataCVA.getEvals())));

		// note that for each subject group, we are averaging the cv scores for each subject 
		// over each condition before storing in pls .mat variables 'designlv', 'b_scores'
		// and 'd_scores'
		int[] classLabels = npairsj.getSetupParams().getClassLabels();
		int[] subjectLabels = npairsj.getSetupParams().getSubjLabels();
		int[] groupLabels = npairsj.getSetupParams().getGroupLabels();
		double[][] fullCVScores = npairsj.fullDataCVA.getCVScores().toArray();
		double[][] avgFullCVScores = avgCVScores(fullCVScores, classLabels, subjectLabels, groupLabels);

		//TODO: design lv should be averaged across subjects (per group?)
		list.add(new MLDouble("designlv", avgFullCVScores));
		list.add(new MLDouble("b_scores", avgFullCVScores));
		list.add(new MLDouble("d_scores", avgFullCVScores));

		// add variables exclusive to npairs
		MLStructure npairs_result = new MLStructure("npairs_result", new int[]{1, 1});
		MLStructure resamp1_result = new MLStructure("resamp1", new int[]{1, 1});
		MLStructure resamp2_result = new MLStructure("resamp2", new int[]{1, 1});
	
		// record volumes (scan no.) used to test each split half as well as
		// subject label associated with each scan no.
		double[][] splitTestVols = getTestVolsAsDbls(npairsj.getSetupParams());
		MLDouble splitTestVolInfo = new MLDouble("split_test_vols", splitTestVols);
		double[][] splitObjLabels = MLFuncs.toDoubleArray(npairsj.getSetupParams().getSplitObjLabels());
		MLDouble splitObjLabInfo = new MLDouble("split_obj_labels", splitObjLabels);
		npairs_result.setField("split_test_vols", splitTestVolInfo);
		npairs_result.setField("split_obj_labels", splitObjLabInfo);
		
		MLStructure predStatsPriors = new MLStructure("priors", new int[]{1, 1});
		MLStructure predStatsNoPriors = new MLStructure("no_priors", new int[]{1, 1});
		
		// rearrange npairs Matrix arrays containing pred stats into 2-element Matrix arrays, one
		// Matrix (dims nAnalyses X maxNTestVols) for priors and the second for no priors
		// NOTE that these Matrices are matlib.Matrices, NOT Jama Matrices.
		// NOTE that MatrixImpl.matlibType must be set outside of this class.
		npairs.shared.matlib.Matrix[] ppTrueClass =  reformPredMats(npairsj.ppTrueClass);
		npairs.shared.matlib.Matrix[] sqrdPredError = reformPredMats(npairsj.sqrdPredError);
		npairs.shared.matlib.Matrix[] correctPred = reformPredMats(npairsj.correctPred);
		npairs.shared.matlib.Matrix[] predClass = reformPredMats(npairsj.predClass);
		
		// rearrange post probs for all classes into structs containing no. classes elems;
		// each elem is 2D array (dims nAnalyses X maxNTestVols)
		int nClasses = npairsj.ppAllClasses[0][0].numCols();
		MLCell ppAllPriors = new MLCell("pp_all_classes", new int[]{1, nClasses});
		MLCell ppAllNoPriors = new MLCell("pp_all_classes", new int[] {1, nClasses});
		int nAnalyses = npairsj.ppAllClasses.length;
		int maxNTestVols = npairsj.ppAllClasses[0][0].numRows();
		
		for (int c = 0; c < nClasses; ++c) {
			npairs.shared.matlib.Matrix ppPriorsCurrCls = 
				new npairs.shared.matlib.MatrixImpl(nAnalyses, maxNTestVols).getMatrix();
			npairs.shared.matlib.Matrix ppNoPriorsCurrCls = 
				new npairs.shared.matlib.MatrixImpl(nAnalyses, maxNTestVols).getMatrix();
			
			for (int a = 0; a < nAnalyses; ++a) {
				ppPriorsCurrCls.setRowQuick(a, npairsj.ppAllClasses[a][0].getColumnQuick(c));
				ppNoPriorsCurrCls.setRowQuick(a, npairsj.ppAllClasses[a][1].getColumnQuick(c));
			}
			ppAllPriors.set(new MLDouble("cls" + c, ppPriorsCurrCls.toArray()), 0, c);
			ppAllNoPriors.set(new MLDouble("cls" + c, ppNoPriorsCurrCls.toArray()), 0, c);
		}
		
		predStatsPriors.setField("pp_true_class", 
				new MLDouble("pp", ppTrueClass[0].toArray()));
		predStatsPriors.setField("sqrd_pred_error", 
				new MLDouble("spe", sqrdPredError[0].toArray()));
	    predStatsPriors.setField("correct_pred", 
	    		new MLDouble("corr_pred", correctPred[0].toArray()));		
		predStatsPriors.setField("pred_class", 
				new MLDouble("pred_cls", predClass[0].toArray()));
		predStatsPriors.setField("pp_all_classes", ppAllPriors);
		
		predStatsNoPriors.setField("pp_true_class", 
				new MLDouble("pp", ppTrueClass[1].toArray()));
		predStatsNoPriors.setField("sqrd_pred_error", 
				new MLDouble("spe", sqrdPredError[1].toArray()));
	    predStatsNoPriors.setField("correct_pred", 
	    		new MLDouble("corr_pred", correctPred[1].toArray())); 
		predStatsNoPriors.setField("pred_class",
				new MLDouble("pred_cls", predClass[1].toArray()));
		predStatsNoPriors.setField("pp_all_classes", ppAllNoPriors);
		
		MLStructure pred_stats = new MLStructure("prediction", new int[]{1, 1});
		pred_stats.setField("priors", predStatsPriors);
		pred_stats.setField("no_priors", predStatsNoPriors);
		

		boolean resamplingDone = npairsj.getSetupParams().resampleData;
		boolean split2ResultsExist = npairsj.getSetupParams().switchTrainAndTestSets;

		// TODO: add option for user to give class names associated with each class label;
		// for now, just name each class using its integer label
		// (Names are given in ascending order of class label)
		int[] uniqClassLabs = MLFuncs.sortAscending(MLFuncs.unique(classLabels));

//		 Save classNames
		MLCell classNames = new MLCell("class_names", new int[]{1, uniqClassLabs.length});
//		if (npairsj.useCondsAsClasses()) {
		for (int i = 0; i < uniqClassLabs.length; i++) { // num labels = num conds
			classNames.set(new MLChar("class_name" + i, condNames[i]), 0, i);
		}
		
		// save full-data r2
		if (npairsj.computeR2) {
			MLDouble r2All = new MLDouble("r2_full_data", npairsj.fullDataCVA.getR2().toArray());
			npairs_result.setField("r2_full_data", r2All);
		}
		
//		}
//		else {
//			for (int i = 0; i < uniqClassLabs.length; i++) {
//				classNames.set(new MLChar("class_name" + i, 
//						Integer.toString(uniqClassLabs[i])), 0, i);
//			}
//		}
		npairs_result.setField("class_names", classNames);
		
		// TODO: Save avg Training/Test CV Scores in appropriately named .mat file 
		//       variables - right now, avg training cv scores are stored in
		//       resamp1.cv_scores_avg and avg test cv scores are stored in 
		//       resamp2.cv_scores_avg. Evals from 1st and 2nd split half analyses
		//       are stored in resamp1 and resamp2, respectively.
		// TODO: implement option to save cv scores, evals for ALL split halves,
		//       not just averages
		if (resamplingDone) {
			String splitObjType = npairsj.getSetupParams().getSplitObjType();
			if (splitObjType.toUpperCase().startsWith("SESSION")) {
				splitObjType = "Session"; // get rid of "(default)" designation in results file
			}
			npairs_result.setField("split_type", new MLChar("split_type", splitObjType));
			npairs_result.setField("num_samples", new MLDouble("num_samples",
					new double[][] {{npairsj.getSetupParams().numSplits}}));
			double[][] avgCVScoresTr = npairsj.avgCVScoresTrain.toArray();
			resamp1_result.setField("cv_scores_avg", new MLDouble("cv_scores", 
					avgCVScoresTr));
			resamp1_result.setField("evals", new MLDouble("evals", 
					MLFuncs.transpose(npairsj.avgSplit1CVAEvals)));
			npairs_result.setField("resamp1", resamp1_result);
			
			// save prediction stats
			npairs_result.setField("prediction", pred_stats);
			// save r2 splits
			if (npairsj.computeR2) {
				int nR2Dim = npairsj.getR2().length; // == no. PCs in CVA analysis
				MLCell r2Splits = new MLCell("r2_splits", new int[] {nR2Dim, 1});
				for (int r = 0; r < nR2Dim; ++r) {
					MLDouble currDimR2 = new MLDouble("currDimR2", npairsj.getR2()[r].toArray());
					r2Splits.set(currDimR2, r);
				}
				npairs_result.setField("r2_splits", r2Splits);
			}
		}
		
		if (split2ResultsExist) {
			double[][] avgCVScoresTest = npairsj.avgCVScoresTest.toArray();
			resamp2_result.setField("cv_scores_avg", new MLDouble("cv_scores", avgCVScoresTest));
			resamp2_result.setField("evals", new MLDouble("evals", 
					MLFuncs.transpose(npairsj.avgSplit2CVAEvals)));
			npairs_result.setField("resamp2", resamp2_result);

			// reproducibility results (exist only if resampling
			// has been done and both splits used as training data)
			double[][] reprodCC = npairsj.getCorrCoeffs().toArray();
			npairs_result.setField("reprod_cc", new MLDouble("reprod_cc", reprodCC));		
		}

		npairs_result.setField("cv_scores", new MLDouble("cv_scores", fullCVScores));
		npairs_result.setField("class_labels", new MLDouble("class_labels", 
				MLFuncs.transpose(MLFuncs.toDoubleArray(classLabels))));

		npairs_result.setField("cv_brainlv_avg", new MLDouble("cv_brainlv_avg", 
				npairsj.avgSpatialPattern.toArray()));
		npairs_result.setField("zscored_brainlv_avg", new MLDouble("zscored_brainlv_avg", 
				npairsj.avgZScorePattern.toArray()));

		list.add(npairs_result);

	}
	

	private double[][] getTestVolsAsDbls(NpairsjSetupParams nsp) throws IOException {
		// read in split vols (training vols for each split half) info from file
		String volsFilename = nsp.resultsFilePrefix + ".vols";
		double[][] tmpSplitVols = NpairsjIO.readFromIDLFile(volsFilename);
		int numSplitHalves = tmpSplitVols.length;
		int numVols = tmpSplitVols[0].length;
		// switch every pair of rows (so each row gives indices of *test* volumes for corresp. split half)
		double[][] splitTestVols = new double[numSplitHalves][numVols];
		for (int i = 0; i < numSplitHalves; i = i + 2) { // loop through even indices only
			splitTestVols[i] = tmpSplitVols[i + 1];
			splitTestVols[i + 1] = tmpSplitVols[i];
		}
		return splitTestVols;
	}

	/** Averages cv scores within group, condition and subject and returns the result
	 *  in a 2D double array with (no. cv dims) number of columns, and rows in the order:
	 *  G1C1S1, G1C1S2, G1C2S1, G1C2S2, G2C1S1, G2C1S2, G2C2S1, G2C2S2
	 *  (note different groups can have diff no. of subjects, but must have same no.
	 *   of classes(conditions))
	 * @param cvScores
	 * @param classLabels
	 * @param subjectLabels
	 * @param groupLabels
	 * @return double array containing averaged results - 1 col per cv dim
	 * @see Analysis.avgCVScores(double[][])
	 */
	private static double[][] avgCVScores(double[][] cvScores, int[] classLabels, 
			int[] subjectLabels, int[] groupLabels) {

		// Note that this code makes no assumptions about ordering of group, class and subj
		// enumerations of elements, hence explicitly rearranges output rows in ascending order
		// of subj within class within group.
		// TODO: is there a better way of ensuring ordering of elements in output data?

		Hashtable<Integer, int[]> groupIndexArrays = CVA.getLabelIndices(groupLabels);
		int nCVDims = cvScores[0].length;

		int nGrps = groupIndexArrays.size();
		int[] uniqGrpLabels = new int[nGrps];
		int gCount = 0;
//		Vector[][] vAvgCVScoresUnordered = null; // will hold cv score avgs before reordering s.t.
		// subj within class within group is in ascending order
		Vector<Double[]> vAvgCVScores = new Vector<Double[]>();
		Vector<Vector> vvAvgCVScores = new Vector<Vector>();

		for (Enumeration<Integer> g = groupIndexArrays.keys(); g.hasMoreElements(); gCount++) {
			Integer currGrpLabel = (Integer)g.nextElement();			
			uniqGrpLabels[gCount] = currGrpLabel.intValue();
			Vector <Double[][]> vAvgCVScoresCurrGrp = new Vector<Double[][]>();

			int[] currGrpIndices = groupIndexArrays.get(currGrpLabel);
			int[] subjLabelsForCurrGrp = MLFuncs.getItemsAtIndices(subjectLabels, 
					currGrpIndices);
			int[] classLabelsForCurrGrp = MLFuncs.getItemsAtIndices(classLabels,
					currGrpIndices);

			Hashtable<Integer, int[]> classIndexArrays = CVA.getLabelIndices(classLabelsForCurrGrp);
			Hashtable<Integer, int[]> subjIndexArrays = CVA.getLabelIndices(subjLabelsForCurrGrp);

			int nClasses = classIndexArrays.size();
//			if (gCount == 0) {
//			vAvgCVScoresUnordered = new Vector[nGrps][nClasses]; // ASSUMPTION: no. of classes 
//			// is same across groups
//			}
			int[] uniqClassLabels = new int[nClasses];
			int nSubj = subjIndexArrays.size();
			int[] uniqSubjLabels = new int[nSubj];

			int cCount = 0;
			for (Enumeration<Integer> c = classIndexArrays.keys(); c.hasMoreElements(); cCount++) {
				Integer currClassLabel = (Integer)c.nextElement();
				uniqClassLabels[cCount] = currClassLabel.intValue();
				// initialize array containing curr. class avgs for each subj
				Double[][] vAvgCVScoresCurrClass = new Double[nSubj][nCVDims];

				int sCount = 0;
				for (Enumeration<Integer> s = subjIndexArrays.keys(); s.hasMoreElements(); sCount++) {
					Integer currSubjLabel = (Integer)s.nextElement();
					uniqSubjLabels[sCount] = currSubjLabel.intValue();

					Vector<Integer> currSubjClassIndices = new Vector<Integer>();
					int[] currSubjIndices = subjIndexArrays.get(currSubjLabel);
					int[] currClassIndices = classIndexArrays.get(currClassLabel);
					for (int nextCurrSubjInd : currSubjIndices) {
						if (MLFuncs.contains(currClassIndices, nextCurrSubjInd)) {
							currSubjClassIndices.add(nextCurrSubjInd);
						}
					} // now have vector of indices corresp. to curr class within curr subj, but
					// indices are relative to subj and class labels within curr group - 
					// hence need to project back into full original index array to recover
					// required cvscore indices
					int[] currSubjClassIndArray = new int[currSubjClassIndices.size()];
					for (int i = 0; i < currSubjClassIndArray.length; ++i) {
						currSubjClassIndArray[i] = currSubjClassIndices.get(i);
					}
					int[] cvCurrSubjClassIndices = MLFuncs.getItemsAtIndices(currGrpIndices, currSubjClassIndArray);
//					System.out.println("Group " + currGrpLabel + " Cond " + currClassLabel
//					+ " Subj " + currSubjLabel + ":");
//					for (int i = 0; i < cvCurrSubjClassIndices.length; ++i) {
//					System.out.print(cvCurrSubjClassIndices[i] + " ");
//					}
//					System.out.println();

					double[][] currSubjClassCVScores = new double[cvCurrSubjClassIndices.length][nCVDims];
					// get relevant rows of input cv scores:
					for (int i = 0; i < currSubjClassCVScores.length; ++i) {
						currSubjClassCVScores[i] = cvScores[cvCurrSubjClassIndices[i]];
					}
					// find mean of each column to get avg cv scores for each cv dim
					Double[] avgCurrSubjClassCVScores = new Double[nCVDims];
					for (int i = 0; i < nCVDims; ++i) {
						avgCurrSubjClassCVScores[i] = 0.0;
					}

					for (int j = 0; j < nCVDims; ++j) {
						for (int i = 0; i < currSubjClassCVScores.length; ++i) {
							avgCurrSubjClassCVScores[j] += 
								currSubjClassCVScores[i][j] / currSubjClassCVScores.length;
						}
					}
					vAvgCVScoresCurrClass[sCount] = avgCurrSubjClassCVScores;
				}
				// reorder output rows so subj in curr class are in ascending order
				int[] sortedSubjIndices = MLFuncs.getSortedIndex(uniqSubjLabels);			
				vAvgCVScoresCurrClass = MLFuncs.sortRows(vAvgCVScoresCurrClass, sortedSubjIndices);
				vAvgCVScoresCurrGrp.add(vAvgCVScoresCurrClass);
			}
			// reorder elements in Vector<Double[][]> so classes in curr grp are in ascending order
			int[] sortedClassIndices = MLFuncs.getSortedIndex(uniqClassLabels);
			vAvgCVScoresCurrGrp = sortElements(vAvgCVScoresCurrGrp, sortedClassIndices);
			vvAvgCVScores.add(vAvgCVScoresCurrGrp);
		}
		// reorder elements in Vector<Vector> so groups are in ascending order
		int[] sortedGroupIndices = MLFuncs.getSortedIndex(uniqGrpLabels);
		vvAvgCVScores = sortElements(vvAvgCVScores, sortedGroupIndices);

		for (int g = 0; g < vvAvgCVScores.size(); ++g) {
			Vector<Double[][]> currGrpScores = vvAvgCVScores.get(g);
			for (int c = 0; c < currGrpScores.size(); ++c) {
				Double[][] currClassScores = currGrpScores.get(c);
				for (int s = 0; s < currClassScores.length; ++s) {
					vAvgCVScores.add(currClassScores[s]);
				}
			}
		}
		double[][] avgCVScores = new double[vAvgCVScores.size()][nCVDims];
		for (int row = 0; row < vAvgCVScores.size(); ++row) {
			Double[] currCVScores = vAvgCVScores.get(row);
			for (int col = 0; col < nCVDims; ++col) {
				avgCVScores[row][col] = currCVScores[col];
			}
		}

		return avgCVScores;
	}
	
	/**
	 * Rearranges given Prediction Stat Matrix array into 2-element array of 
	 * Matrices (dims nAnalyses X nVols).  Rows of 1st Matrix are the 1st
	 * rows in each input Matrix; rows of 2nd Matrix are the 2nd rows in
	 * each input Matrix. 
	 * 
	 * NOTE that these are NOT Jama Matrices but matlib.Matrices
	 * 
	 * REQUIRED: Input Matrices all have 2 rows and the same number of columns
	 * REQUIRED: MatrixImpl.matlibType is set outside of this class.
	 * 
	 * @param npairs.shared.matlib.Matrix[] mats
	 *               array (length nAnalyses) of Matrices with dims 2 X nVols
	 */
	private npairs.shared.matlib.Matrix[] reformPredMats(
			npairs.shared.matlib.Matrix[] mats) {
		int nAnalyses = mats.length;
		int nVols = mats[0].numCols();
		npairs.shared.matlib.Matrix[] newMats = new npairs.shared.matlib.Matrix[2];
		
		for (int i = 0; i < 2; ++i) {
			newMats[i] = 
				new npairs.shared.matlib.MatrixImpl(nAnalyses, nVols).getMatrix();
			for (int a = 0; a < nAnalyses; ++a) {
				newMats[i].setRow(a, mats[a].getRow(i));
			}
		}
		return newMats;
	}

	//TODO: parameterize for any type Vector and probably move out of this class
	/** Returns Vector with elements reordered in order indicated by 
	 *  input int array of indices.
	 *  
	 */
	private static Vector sortElements(Vector data, int[] sortedIndices) {
		int nElements = data.size();
		Vector tmpSortedData = new Vector();
		for (int i = 0; i < nElements; ++i) {
			tmpSortedData.add(data.get(sortedIndices[i]));
		}
		data = tmpSortedData;
		return data;
	}

	

	private void addTaskPlsResult(ComputeNonrotatedTaskPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("designlv", mainPls.designLV.getArray()));
		list.add(new MLDouble("b_scores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("d_scores", mainPls.designScores.getArray()));
		list.add(new MLDouble("lvintercorrs", mainPls.LVInterCorrs.getArray()));
	}

	private void addMultiblockPlsResult(ComputeMultiblockPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("designlv", mainPls.designLV.getArray()));
		list.add(new MLDouble("behavlv", mainPls.behavLV.getArray()));
		list.add(new MLDouble("brainscores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("b_scores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("d_scores", mainPls.designScores.getArray()));
		list.add(new MLDouble("lvcorrs", mainPls.lvCorrs.getArray()));

		if(mainPls.origPost != null) {
			list.add(new MLDouble("origpost", mainPls.origPost.getArray()));
		} else {
			list.add(new MLDouble("origpost", new int[]{0,0}));
		}

		MLCell datamatcorrs_lst = new MLCell("datamatcorrs_lst", new int[]{1, mainPls.datamatCorrsList.size()});
		for(int i = 0; i < mainPls.datamatCorrsList.size(); i++) {
			datamatcorrs_lst.set(new MLDouble(null, mainPls.datamatCorrsList.get(i).getArray()), 0, i);
		}
		list.add(datamatcorrs_lst);
	}
	private void addDeviationPETPlsResult(ComputeBehaviorPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("designlv", mainPls.behavLV.getArray()));//behavLV is designLV in TaskPLS
		list.add(new MLDouble("brainscores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("designscores", mainPls.behavScores.getArray()));// designscore is designLV in TaskPLS
	}
	private void addBehaviorPlsResult(ComputeBehaviorPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("behavlv", mainPls.behavLV.getArray()));
		list.add(new MLDouble("brainscores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("b_scores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("lvcorrs", mainPls.lvCorrs.getArray()));

		if(mainPls.origPost != null) {
			list.add(new MLDouble("origpost", mainPls.origPost.getArray()));
		} else {
			list.add(new MLDouble("origpost", new int[]{0,0}));
		}

		MLCell datamatcorrs_lst = new MLCell("datamatcorrs_lst", new int[]{1, mainPls.datamatCorrsList.size()});
		for(int i = 0; i < mainPls.datamatCorrsList.size(); i++) {
			datamatcorrs_lst.set(new MLDouble(null, mainPls.datamatCorrsList.get(i).getArray()), 0, i);
		}
		list.add(datamatcorrs_lst);
	}
	
	private void addNonRotatedBehavPlsResult(ComputeNonRotatedBehavPls mainPls) {
		list.add(new MLDouble("brainlv", mainPls.brainLV.getArray()));
		list.add(new MLDouble("s", MLFuncs.transpose(mainPls.S.getRowPackedCopy())));
		list.add(new MLDouble("behavlv", mainPls.behavLV.getArray()));
		list.add(new MLDouble("brainscores", mainPls.brainScores.getArray()));
		list.add(new MLDouble("behavscores", mainPls.behavScores.getArray()));
		list.add(new MLDouble("lvcorrs", mainPls.lvCorrs.getArray()));
		list.add(new MLDouble("lvintercorrs", mainPls.LVInterCorrs.getArray()));

		MLCell datamatcorrs_lst = new MLCell("datamatcorrs_lst", new int[]{1, mainPls.datamatCorrsList.size()});
		for(int i = 0; i < mainPls.datamatCorrsList.size(); i++) {
			datamatcorrs_lst.set(new MLDouble(null, mainPls.datamatCorrsList.get(i).getArray()), 0, i);
		}
		list.add(datamatcorrs_lst);
	}


	private void addLvEventList(int[] lvEventList) {
		if(lvEventList != null) {
			list.add(new MLDouble("lv_evt_list", MLFuncs.toDoubleArray(lvEventList)));
		}
	}

	private void addDesign(Matrix design) {
		list.add(new MLDouble("design", design.getArray()));
	}
	
	public static void main(String[] args) {
		// test avgCVScores(double[][] int[] int[] int[]):
		int[] classLabels = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 
				1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 
				1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
				1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2};
		int[] groupLabels = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
		int[] subjLabels  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
		int numScans = classLabels.length;
		int numCVDims = 3;

		double[][] origCVScores = new double[numScans][numCVDims];
		for (int i = 0; i < numScans; ++i) {
			for (int j = 0; j < numCVDims; ++j) {
				origCVScores[i][j] = Math.random();
			}
		}

		double[][] avgCVScores = avgCVScores(origCVScores, classLabels, subjLabels, groupLabels);

		npairs.shared.matlib.ColtMatrix origCVScoresMat = new npairs.shared.matlib.ColtMatrix(origCVScores);
		System.out.println("Original input CVScores: ");
		origCVScoresMat.print();
		origCVScoresMat.printToFile("/haier/anita/PLS/testOrigCVScores.idlMat", "IDL");
//		for (int i = 0; i < numScans; ++i) {
//		for (int j = 0; j < numCVDims; ++j) {
//		System.out.print(origCVScores[i][j] + " ");
//		}
//		System.out.println();
//		}
		npairs.shared.matlib.ColtMatrix avgCVScoresMat = new npairs.shared.matlib.ColtMatrix(avgCVScores);
		System.out.println("Averaged CVScores: ");
//		for (int i = 0; i < avgCVScores[0].length; ++i) {
//		for (int j = 0; j < numCVDims; ++j) {
//		System.out.print(avgCVScores[i][j] + " ");
//		}
//		System.out.println();
//		} 
		avgCVScoresMat.print();
		avgCVScoresMat.printToFile("/haier/anita/PLS/testAvgCVScores.idlMat", "IDL");
	}
}

