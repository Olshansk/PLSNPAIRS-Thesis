package pls.analysis;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;

/**
 * Class to perform PLS Analysis on fMRI data using Multiblock approach.
 * @author imran
 *
 */
public class ComputeMultiblockPlsMain {
	/**
	 * Runs the analysis on the given Parameters.
	 * @throws Exception
	 */
	//constructor for fMRI
	public ComputeMultiblockPlsMain(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			double confidenceLevel, String behaviorFilename, String resultsFile, 
			StreamedProgressHelper progress) throws Exception {
		
		RRIBootstrapCheck bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, 
				numBootstraps, false, 50.0);
		
		int[] bscan = MLFuncs.range(0, st.numConditions - 1);

		progress.startTask("Computing reference multiblock PLS", "Ref. multiblock PLS");
		ComputeMultiblockPls mainPls = new ComputeMultiblockPls(st.datamat, st.behavData, st.posthocData, 
				st.behavDataList, st.newDataList, st.numSubjectList, st.numConditions, bscan);
		progress.endTask(); // ref. multiblock PLS
		
		PermutationResult permResult = new ComputeMultiblockPermutationPls(mainPls, st.datamat, 
				st.posthocData, st.numConditions, numPermutations, st.subjectGroup, 
				bscan, progress).permResult;
		
		BootstrapResult bootResult = new ComputeMultiblockBootstrapPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, 
				bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, 
				bscan, confidenceLevel, progress).bootResult;

		progress.startTask("Saving the results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, confidenceLevel, behaviorFilename, resultsFile, 
				permResult, mainPls, bootResult, bscan);
		progress.endTask(); // save results
	}
	
	
	//Constructor for PET
	//it also works for fMRI
	public ComputeMultiblockPlsMain(int imagingType, ConcatenateDatamat st, int numBootstraps, 
			int numPermutations, double confidenceLevel, String behaviorFilename, String resultsFile, 
			StreamedProgressHelper progress) throws Exception {
		RRIBootstrapCheck bootCheck;
		if(imagingType == 2){
			bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.num_cond_lst[0], numBootstraps, 
					false, 50.0);
		}
		else{
			bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, numBootstraps, 
					false, 50.0);
	    }
		int[] bscan = MLFuncs.range(0, st.numConditions - 1);

		progress.startTask("Computing reference multiblock PLS", "Ref. multiblock PLS");
		
		//old calling for fMRI only
		//ComputeMultiblockPls mainPls = new ComputeMultiblockPls(imagingType, st.datamat, st.behavData, 
		//st.posthocData, st.behavDataList, st.newDataList, st.numSubjectList, st.numConditions, bscan);

		//new calling for fMRI and PET
		ComputeMultiblockPls mainPls = new ComputeMultiblockPls(imagingType, st, bscan);
		progress.endTask(); // ref. multiblock PLS
		
		PermutationResult permResult = new ComputeMultiblockPermutationPls(imagingType, mainPls, st, 
				numPermutations, bscan, progress).permResult;
		
		//old calling for only fMRI
		//BootstrapResult bootResult = new ComputeMultiblockBootstrapPls(mainPls, st.datamat, 
		//st.numConditions, st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, 
		//bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, bscan, 
		//confidenceLevel, progress).bootResult;
		
		//new Call for fMRI and PET
		BootstrapResult bootResult = new ComputeMultiblockBootstrapPls(imagingType, mainPls, st, 
				numBootstraps, bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, 
				bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, bscan, 
				confidenceLevel, progress).bootResult;

		progress.startTask("Saving the results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, confidenceLevel, behaviorFilename, 
				resultsFile, permResult, mainPls, bootResult, bscan);
 		progress.endTask(); // save results
	}
}