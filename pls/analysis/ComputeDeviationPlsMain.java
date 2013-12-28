package pls.analysis;

import pls.shared.StreamedProgressHelper;

/**
 * Class to perform PLS Analysis on fMRI data using Mean Centering approach.
 * @author imran
 *
 */
public class ComputeDeviationPlsMain {
	/**
	 * Runs the analysis on the given Parameters.
	 * @param par the parameters to perform the analysis on
	 * @throws Exception
	 */
	public ComputeDeviationPlsMain(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			String resultsFile, StreamedProgressHelper progress) throws Exception {
		// Check to see if the number of bootstraps selected is possible.
		// TODO: This should prompt for new number of bootstraps if necessary when gui is implemented
		RRIBootstrapCheck bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, numBootstraps, 
				true, 50.0);
		
		progress.startTask("Computing reference mean-centering PLS", "Ref. mean-centering PLS");
		ComputeDeviationPls mainPls = new ComputeDeviationPls(st.datamat, st.numConditions, st.eventList, 
				st.subjectGroup);
		progress.endTask(); // ref. mean-centring PLS
	
		PermutationResult permResult = new ComputeDeviationPermutationPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numPermutations, st.subjectGroup, progress).permResult;
		
		// Run the bootstraps
		BootstrapResult bootResult = new ComputeDeviationBootstrapPls(mainPls, st.datamat, st.numConditions, 
				st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, 
				bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, 
				progress).bootResult;

		progress.startTask("Saving the results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, resultsFile, permResult, mainPls, bootResult);
		progress.endTask(); // save results
	}
}
