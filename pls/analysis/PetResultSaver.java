package pls.analysis;

import java.util.ArrayList;
import java.util.Date;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLStructure;
import java.io.*;

import Jama.Matrix;

public class PetResultSaver {
	
	private ArrayList<MLArray> list = new ArrayList<MLArray>();
	
	// Deviation
	public PetResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, String resultsFile, PermutationResult permResult, ComputeDeviationPls mainPls, BootstrapResult bootResult) throws Exception {
		addStObjects(st);
        
        list.add(new MLChar("ContrastFile", "NONE"));

        list.add(new MLChar("create_ver", "999999"));
        
        addPermutationResult(permResult);
        
        addBootstrapResult(bootResult);
        
        addDeviationPlsResult(mainPls);
        
        addLvEventList(st.eventList);
        
        new MatFileWriter(resultsFile, list);
	}
	
	// Nonrotated task
	public PetResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, String contrastFile, String resultsFile, PermutationResult permResult, ComputeNonrotatedTaskPls mainPls, BootstrapResult bootResult, Matrix design) throws Exception {
		addStObjects(st);
        
        list.add(new MLChar("ContrastFile", contrastFile));

        list.add(new MLChar("create_ver", "999999"));
        
        addPermutationResult(permResult);
        
        addBootstrapResult(bootResult);
        
        addTaskPlsResult(mainPls);
        
        addLvEventList(st.eventList);
        
        addDesign(design);
        
        new MatFileWriter(resultsFile, list);
	}
	
	// Multiblock
	public PetResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, double confidenceLevel, String behaviorFile, String resultsFile, PermutationResult permResult, ComputeMultiblockPls mainPls, BootstrapResult bootResult, int[] bscan) throws Exception {
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
	
	// Behavior
	public PetResultSaver(ConcatenateDatamat st, int numBootstraps, int numPermutations, double confidenceLevel, String behaviorFile, String resultsFile, PermutationResult permResult, ComputeBehaviorPls mainPls, BootstrapResult bootResult) throws Exception {
		addStObjects(st);
        
        list.add(new MLChar("ContrastFile", "NONE"));

        list.add(new MLChar("create_ver", "999999"));
        
        addPermutationResult(permResult);
        
        addBootstrapResult(bootResult);
        
        addBehaviorPlsResult(mainPls);
        
        addLvEventList(st.eventList);
        
        new MatFileWriter(resultsFile, list);
	}
	
	private void addStObjects(ConcatenateDatamat st) {
        // Save datamat Profiles

		MLCell sp = new MLCell("datamat_files", new int[]{1, st.datamatProfiles.size()});
        for(int i = 0; i < st.datamatProfiles.size(); i++) {
    		MLCell currSp = new MLCell(null, new int[]{st.datamatProfiles.get(i).length, 1});
        	for(int j = 0; j < st.datamatProfiles.get(i).length; j++) {
            	currSp.set(new MLChar(null, st.datamatProfiles.get(i)[j]), j, 0);
        	}
        	sp.set(currSp, 0, i);
        }
        MLCell spt = new MLCell("datamat_files_timestamp", new int[]{1, st.datamatProfiles.size()});
        for(int i = 0; i < st.datamatProfiles.size(); i++) {
        	MLCell currSp = new MLCell(null, new int[]{st.datamatProfiles.get(i).length, 1});
        	for(int j = 0; j < st.datamatProfiles.get(i).length; j++) {
        		File filen = new File(st.datamatProfiles.get(i)[0]);
            	long lastmodified = filen.lastModified();
            	Date d = new Date(lastmodified);
            	String nd = d.getDate()+":"+d.getTime();
            	currSp.set(new MLChar(null, nd), j, 0);
            	System.out.println("Date is"+ nd);
        	}
        	spt.set(currSp, 0, i);
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
        
        //list.add(new MLDouble("cond_selection", MLFuncs.toDoubleArray(st.conditionSelection)));
        list.add(new MLDouble("new_coords", MLFuncs.toDoubleArray(st.coords)));
        list.add(new MLDouble("dims", MLFuncs.toDoubleArray(st.dims)));
        list.add(new MLDouble("num_cond_lst", new double[][]{{st.numConditions}}));
        
        //list.add(new MLDouble("st_win_size", new double[][]{{st.winSize}}));
        list.add(new MLDouble("voxel_size", MLFuncs.to2DArray(st.voxelSize)));
        list.add(new MLDouble("origin", MLFuncs.toDoubleArray(st.origin)));
        list.add(new MLDouble("subj_group", MLFuncs.toDoubleArray(st.subjectGroup)));
        
        // Save subjectName lst
        MLCell subj_name = new MLCell("subj_name_lst", new int[]{1, st.subjectName.size()});
        for(int i = 0; i < st.subjectName.size(); i++) {
        	subj_name.set(new MLChar("subj_name_lst" + i, st.subjectName.get(i)), 0, i);
        }
        
        list.add(subj_name);
        /*
        // Save conditionNames
        MLCell cond_name = new MLCell("cond_name", new int[]{1, st.conditions.length});
        for(int i = 0; i < st.conditions.length; i++) {
        	cond_name.set(new MLChar("cond_name" + i, st.conditions[i]), 0, i);
        }
        list.add(cond_name);
        */
        list.add(new MLDouble("num_subj_lst", MLFuncs.toDoubleArray(st.numSubjectList)));
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
        		Double[] dFlattened = new Double[flattened.length]; 
        		for(int i = 0; i < flattened.length ; i++) {
        			dFlattened[i] = new Double(flattened[i]);
        		}
        		MLDouble distrib = new MLDouble("distrib", new int[]{bootResult.distrib[0].getRowDimension(), bootResult.distrib[0].getColumnDimension(), bootResult.distrib.length});
        		distrib.setReal(dFlattened);
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
	
	private void addLvEventList(int[] lvEventList) {
        if(lvEventList != null) {
            list.add(new MLDouble("lv_evt_list", MLFuncs.toDoubleArray(lvEventList)));
        }
	}
	
	private void addDesign(Matrix design) {
        list.add(new MLDouble("design", design.getArray()));
	}
}
