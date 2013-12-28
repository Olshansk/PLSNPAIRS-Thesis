package pls.analysis;

import Jama.Matrix;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;

public class ComputeNonRotatedTaskPermutationPls {
	
	protected PermutationResult permResult = null;

	//constructor for fMRI
	public ComputeNonRotatedTaskPermutationPls(ComputeNonrotatedTaskPls refPls, Matrix stDatamat, 
			int numConditions, int[] eventList, int numPermutations, int[] subjectGroup, Matrix design, 
			StreamedProgressHelper progress) {
		
		if(numPermutations == 0) {
			return;
		}
		
		// Generate the permutation orders
		int[][] permOrder = new RRIPermutationOrder(subjectGroup, numConditions, numPermutations).result;
		
		Matrix permCount = new Matrix(refPls.S.getRowDimension(), 1);
		
		for(int k = 0; k < numPermutations; k++) {
			progress.startTask("Computing permutation no. " + (k + 1), "Perm. no. " + (k + 1));
			int[] newOrder = MLFuncs.getColumn(permOrder, k);

			GroupDeviationData dev = new GroupDeviationData(MLFuncs.getRows(stDatamat, newOrder), 
					numConditions, eventList, subjectGroup, design);

			Matrix crossBlock = MLFuncs.normalizeEuc(dev.design, 1).transpose().times(dev.data);
			
			Matrix S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
			
			permCount.plusEquals(MLFuncs.greaterThanOrEqualTo(S, refPls.S));
			progress.endTask(); // curr. permutation

		}
		
		permResult = new PermutationResult();
		permResult.sProb = permCount.times(1.0 / numPermutations);
		permResult.numPermutations = numPermutations;
		permResult.permSample = permOrder;
		permResult.sp = permCount;
		permResult.dp = null;
		permResult.designLVprob = null;
	}
	
	//constructor for only PET
	public ComputeNonRotatedTaskPermutationPls(int imagingType, ComputeNonrotatedTaskPls refPls, 
			ConcatenateDatamat st, int numPermutations, Matrix design, StreamedProgressHelper progress) {
		
		if(numPermutations == 0) {
			return;
		}
		
		// Generate the permutation orders
		int[][] permOrder = new RRIPermutationOrder(st.numSubjectList, st.num_cond_lst[0], 
				numPermutations).result;
		
		Matrix permCount = new Matrix(refPls.S.getRowDimension(), 1);
		
		for(int k = 0; k < numPermutations; k++) {
			progress.startTask("Computing permutation number " + (k + 1), "Perm. no. " + (k + 1));
			
			int[] newOrder = MLFuncs.getColumn(permOrder, k);
			
			Matrix data_p = MLFuncs.getRows(st.datamat,newOrder);
			
			GroupDeviationData dev = new GroupDeviationData(st, data_p, design);
		
			Matrix crossBlock = MLFuncs.normalizeEuc(dev.design, 1).transpose().times(dev.data);
			
			Matrix S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
			
			permCount.plusEquals(MLFuncs.greaterThanOrEqualTo(S, refPls.S));
			progress.endTask(); // curr. permutation
		}
		
		permResult = new PermutationResult();
		permResult.sProb = permCount.times(1.0 / numPermutations);
		permResult.numPermutations = numPermutations;
		permResult.permSample = permOrder;
		permResult.sp = permCount;
		permResult.dp = null;
		permResult.designLVprob = null;
	}
}
