package pls.analysis;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import Jama.Matrix;

public class ComputeNonRotatedBehavPermutationPls {
	protected PermutationResult permResult = null;

	//constructor for fMRI
	public ComputeNonRotatedBehavPermutationPls(ComputeNonRotatedBehavPls refPls, Matrix stDatamat, 
			int numConditions, int[] eventList, int numPermutations, int[] numSubjectList, int[] subjectGroup, Matrix design, 
			StreamedProgressHelper progress) throws Exception{
		
		int numGroups = subjectGroup.length;
		
		if(numPermutations == 0) {
			return;
		}
		
		// Generate the permutation orders
		int[][] permOrder = new RRIPermutationOrder(subjectGroup, numConditions, numPermutations).result;
		
		Matrix permCount = new Matrix(refPls.S.getRowDimension(), 1);
		
		for(int p = 0; p < numPermutations; p++) {
			progress.startTask("Computing permutation no. " + (p + 1), "Perm. no. " + (p + 1));
			
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(permOrder, p));
			Matrix stackedData = null;
			for(int g = 1; g <= numGroups; g++) {
				int n = numSubjectList[g - 1];

				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, 
						MLFuncs.range(0, g - 2))) * numConditions;

				int[] range = MLFuncs.range(span, n * numConditions + span - 1);

				// Check for upcoming NaN and re-sample if necessary.  This only happened on 
				// behavior analysis, because the 'xcor' inside of 'rri_corr_maps' contains a 
				// 'stdev', which is a divident. If it is 0, it will cause divided by 0 problem.
				// Since this happend very rarely, so the speed will not be affected that much.
				double min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
				int count = 0;

				while(min1 == 0) {
					permOrder = MLFuncs.setColumn(permOrder, p, MLFuncs.randomPermutations(stDatamat.
							getRowDimension()));
					behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(permOrder, p));
					min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
					count++;

					if(count > 100) {
						throw new Exception("Please check your behavior data, and make sure " +
						"none of the columns are all the same for each group");
					}
				}

				Matrix data = null;

				// Notice here that stacked_datamat is used, instead of boot_p. This is only for behavpls_perm.
				if(numGroups == 1) {
					data = new RRICorrMaps(behavP, stDatamat, n, numConditions).maps;
				} else {
					data = new RRICorrMaps(MLFuncs.getRows(behavP, range), MLFuncs.getRows(stDatamat, 
							range), n, numConditions).maps;
				}

				stackedData = MLFuncs.append(stackedData, data);
			}

			Matrix crossBlock = MLFuncs.normalizeEuc(design, 1).transpose().times(stackedData);
			Matrix S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
			
			permCount.plusEquals(MLFuncs.greaterThanOrEqualTo(S, refPls.S));
			progress.endTask(); // curr. permutation

		}
		
		permResult = new PermutationResult();
		permResult.sProb = permCount.times(1.0 / numPermutations);
		permResult.numPermutations = numPermutations;
		permResult.permSample = permOrder;
		permResult.sp = permCount;
	
	}
}
