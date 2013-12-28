package pls.analysis;

import pls.shared.StreamedProgressHelper;

/**
 * Class to perform PLS Analysis on fMRI data using behavior approach.
 * @author imran
 *
 */
public class ComputeBehaviorPlsMain {
	/**
	 * Runs the analysis on the given Parameters.
	 * @throws Exception
	 */
	//constructor for fMRI
	public ComputeBehaviorPlsMain(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			double confidenceLevel, String behaviorFilename, String resultsFile, 
			StreamedProgressHelper progress) throws Exception {
		
		RRIBootstrapCheck bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, 
				numBootstraps, false, 50.0);


		progress.startTask("Computing reference behavior PLS...", "Ref. behavior PLS");
		ComputeBehaviorPls mainPls = new ComputeBehaviorPls(st.datamat, st.behavData,  
				st.behavDataList, st.newDataList, st.numSubjectList, st.numConditions);
		progress.endTask(); // ref. behav. PLS

		
		PermutationResult permResult = new ComputeBehaviorPermutationPls(mainPls, st.datamat, 
				st.numConditions, numPermutations, st.subjectGroup, progress).permResult;
		
		BootstrapResult bootResult = new ComputeBehaviorBootstrapPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, confidenceLevel, progress).bootResult;


		progress.startTask("Saving results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, confidenceLevel, behaviorFilename, 
				resultsFile, permResult, mainPls, bootResult);
		progress.endTask();

	}
	//constructor for PET and fMRI
	public ComputeBehaviorPlsMain(int imagingType, ConcatenateDatamat st, int numBootstraps, 
			int numPermutations, double confidenceLevel, String behaviorFilename, 
			String resultsFile, boolean isbehav, StreamedProgressHelper progress) throws Exception {
		
		System.out.println("stsubjectGropup "+st.subjectGroup);

		RRIBootstrapCheck bootCheck;
		
		if(imagingType == 2)
			bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.num_cond_lst[0], numBootstraps, 
					false, 50.0);
		else
			bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, numBootstraps, 
					false, 50.0);
				
		progress.startTask("Computing reference behavior PLS...", "Ref. behavior PLS");	
		ComputeBehaviorPls mainPls = new ComputeBehaviorPls(imagingType, st, isbehav);	
		progress.endTask(); // ref. behav. PLS
		
		PermutationResult permResult = new ComputeBehaviorPermutationPls(imagingType, mainPls, 
				st, numPermutations, progress, isbehav).permResult;
		
		BootstrapResult bootResult =null;
		if(imagingType == 2 && isbehav==false) // task PLS
			bootResult = new ComputeBehaviorBootstrapPls(st,mainPls, numBootstraps, 
					bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, 
					bootCheck.newNumBootstraps, confidenceLevel, progress, false).bootResult;
		else if(imagingType == 2 && isbehav==true) //behavior 
			bootResult = new ComputeBehaviorBootstrapPls( st, mainPls,numBootstraps, 
					bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, 
					bootCheck.newNumBootstraps, confidenceLevel, progress, true).bootResult;
		else
			bootResult = new ComputeBehaviorBootstrapPls(mainPls, st.datamat, st.numConditions, 
					st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, 
					bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, 
					confidenceLevel, progress).bootResult;
		
		progress.startTask("Saving the results...", "Saving results");
		if(imagingType == 2 && isbehav==false) // task PLS
			new ResultSaver(imagingType, st, numBootstraps, numPermutations, resultsFile, permResult, 
					mainPls, bootResult);
		else if(imagingType == 2 && isbehav==true) //behavior 
			new ResultSaver(imagingType, st, numBootstraps, numPermutations, confidenceLevel, 
					behaviorFilename, resultsFile, permResult, mainPls, bootResult);
		else //fMRI
			new ResultSaver(st, numBootstraps, numPermutations, confidenceLevel, behaviorFilename, 
					resultsFile, permResult, mainPls, bootResult);
		progress.endTask(); // saving the results
	}
}