package pls.analysis;

import Jama.Matrix;
import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;

public class ComputeNonRotatedBehavPlsMain {
	
	//constructor for fMRI
	public ComputeNonRotatedBehavPlsMain(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			double confidenceLevel, String behaviorFilename, String contrastFile, String resultsFile, 
			StreamedProgressHelper progress) throws Exception {
		
		Matrix design = new Matrix(MLFuncs.load(contrastFile));
		
		RRIBootstrapCheck bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, 
				numBootstraps, false, 50.0);


		progress.startTask("Computing reference behavior PLS...", "Ref. behavior PLS");
		ComputeNonRotatedBehavPls mainPls = new ComputeNonRotatedBehavPls(st.datamat, st.behavData, 
				st.behavDataList, st.newDataList, st.numSubjectList, st.numConditions, st.eventList, st.subjectGroup, design);
		progress.endTask(); // ref. behav. PLS
		
		PermutationResult permResult = new ComputeNonRotatedBehavPermutationPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numPermutations, st.numSubjectList, st.subjectGroup, design, progress).permResult;
		
		BootstrapResult bootResult = new ComputeNonRotatedBehavBootstrapPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numBootstraps, st.numSubjectList, st.subjectGroup, 
				bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, design, progress).bootResult;

		progress.startTask("Saving results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, confidenceLevel, behaviorFilename, 
				contrastFile, resultsFile, permResult, mainPls, bootResult, design);
		progress.endTask();

	}
	//constructor for PET
	
}
