package pls.analysis;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import Jama.Matrix;

/**
 * Class to perform PLS Analysis on fMRI data using non-rotated task approach.
 * @author imran
 *
 */
public class ComputeNonRotatedTaskPlsMain {
	/**
	 * Runs the analysis on the given Parameters.
	 * @param par the parameters to perform the analysis on
	 * @throws Exception
	 */
	//constructor for fMRI
	public ComputeNonRotatedTaskPlsMain(ConcatenateDatamat st, int numBootstraps, int numPermutations, 
			String contrastFile, String resultsFile, StreamedProgressHelper progress) throws Exception {
		
		Matrix design = new Matrix(MLFuncs.load(contrastFile));
		
		RRIBootstrapCheck bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, 
				numBootstraps, false, 50.0);
		
		progress.startTask("Computing reference non-rotated task PLS...", "Ref. non-rotated task PLS");
		ComputeNonrotatedTaskPls mainPls = new ComputeNonrotatedTaskPls(st.datamat, st.numConditions, 
				st.eventList, st.subjectGroup, design);
		progress.endTask(); // ref. non-rot. PLS

		PermutationResult permResult = new ComputeNonRotatedTaskPermutationPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numPermutations, st.subjectGroup, 
				design, progress).permResult;
		
		BootstrapResult bootResult = new ComputeNonRotatedTaskBootstrapPls(mainPls, st.datamat, 
				st.numConditions, st.eventList, numBootstraps, st.subjectGroup, 
				bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, 
				bootCheck.newNumBootstraps, design, progress).bootResult;
		
		progress.startTask("Saving the results...", "Saving results");
		new ResultSaver(st, numBootstraps, numPermutations, contrastFile, resultsFile, permResult, 
				mainPls, bootResult, design);
		progress.endTask(); // save results
	}
	
	
	//constructor for fMRI and PET
	public ComputeNonRotatedTaskPlsMain(int imagingType, ConcatenateDatamat st, int numBootstraps, 
			int numPermutations, String contrastFile, String resultsFile, StreamedProgressHelper progress) 
		throws Exception {
		
		Matrix design = new Matrix(MLFuncs.load(contrastFile));
	
		RRIBootstrapCheck bootCheck =null;

		if(imagingType == 2)
			 bootCheck = new RRIBootstrapCheck(st.numSubjectList, st.num_cond_lst[0], numBootstraps, 
					 false, 50.0);
		else			
			bootCheck = new RRIBootstrapCheck(st.subjectGroup, st.numConditions, numBootstraps, 
					false, 50.0);
		
		progress.startTask("Computing reference non-rotated task PLS", "Ref. non-rotated task PLS");
		ComputeNonrotatedTaskPls mainPls = null;
		
		if(imagingType == 2)
			mainPls = new ComputeNonrotatedTaskPls(st, design);
		else
			mainPls = new ComputeNonrotatedTaskPls(st.datamat, st.numConditions, st.eventList, 
					st.subjectGroup, design);
		
		progress.endTask(); // ref. PLS
		
		PermutationResult permResult=null;
		BootstrapResult bootResult = null;
		if(imagingType == 2){
			permResult = new ComputeNonRotatedTaskPermutationPls(imagingType, mainPls, st, 
					numPermutations, design, progress).permResult;
			bootResult = new ComputeNonRotatedTaskBootstrapPls(st, mainPls, numBootstraps, 
					bootCheck.minSubjectsPerGroup, bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, 
					bootCheck.newNumBootstraps, design, progress).bootResult;		
		}else{	
			permResult = new ComputeNonRotatedTaskPermutationPls(mainPls, st.datamat, st.numConditions, 
					st.eventList, numPermutations, st.subjectGroup, design, progress).permResult;
			bootResult = new ComputeNonRotatedTaskBootstrapPls(mainPls, st.datamat, st.numConditions, 
					st.eventList, numBootstraps, st.subjectGroup, bootCheck.minSubjectsPerGroup, 
					bootCheck.isBootstrapSamples, bootCheck.bootstrapSamples, bootCheck.newNumBootstraps, 
					design, progress).bootResult;
		}

		progress.startTask("Saving the results...", "Saving results");
		if(imagingType == 2)
			new ResultSaver(imagingType, st, numBootstraps, numPermutations, contrastFile, resultsFile, 
					permResult, mainPls, bootResult, design);
		else
			new ResultSaver(st, numBootstraps, numPermutations, contrastFile, resultsFile, permResult, 
					mainPls, bootResult, design);
		progress.endTask(); //save results
	}
}