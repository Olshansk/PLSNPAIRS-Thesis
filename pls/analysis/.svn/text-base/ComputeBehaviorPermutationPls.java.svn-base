package pls.analysis;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputeBehaviorPermutationPls {

	protected PermutationResult permResult = null;

	/* constructor for fMRI */ 
	public ComputeBehaviorPermutationPls(ComputeBehaviorPls refPls, Matrix stackedDatamat, 
			int k, int numPermutations, int[] numSubjectList, 
			StreamedProgressHelper progress) throws Exception {
		int numGroups = numSubjectList.length;

		Matrix sp = new Matrix(refPls.S.getRowDimension(), 1);
		Matrix dp = new Matrix(refPls.behavLV.getRowDimension(), refPls.behavLV.getColumnDimension());

		int[][] reorder = new int[stackedDatamat.getRowDimension()][numPermutations];

		for(int p = 0; p < numPermutations; p++) {
			reorder = MLFuncs.setColumn(reorder, p, MLFuncs.randomPermutations(stackedDatamat.
					getRowDimension()));
		}

		for(int p = 0; p < numPermutations; p++) {
			progress.startTask("Computing permutation no. " + (p + 1), "Perm. no. " + (p + 1));
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(reorder, p));
			Matrix stackedData = null;
			for(int g = 1; g <= numGroups; g++) {
				int n = numSubjectList[g - 1];

				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, 
						MLFuncs.range(0, g - 2))) * k;

				int[] range = MLFuncs.range(span, n * k + span - 1);

				// Check for upcoming NaN and re-sample if necessary.  This only happened on 
				// behavior analysis, because the 'xcor' inside of 'rri_corr_maps' contains a 
				// 'stdev', which is a divident. If it is 0, it will cause divided by 0 problem.
				// Since this happend very rarely, so the speed will not be affected that much.
				double min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
				int count = 0;

				while(min1 == 0) {
					reorder = MLFuncs.setColumn(reorder, p, MLFuncs.randomPermutations(stackedDatamat.
							getRowDimension()));
					behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(reorder, p));
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
					data = new RRICorrMaps(behavP, stackedDatamat, n, k).maps;
				} else {
					data = new RRICorrMaps(MLFuncs.getRows(behavP, range), MLFuncs.getRows(stackedDatamat, 
							range), n, k).maps;
				}

				stackedData = MLFuncs.append(stackedData, data);
			}
			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());

			Matrix sPerm = USV.getS();
			Matrix pBehavLV = USV.getV();

			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.behavLV, pBehavLV).rotatedMatrix;
			pBehavLV = pBehavLV.times(sPerm).times(rotatedMatrix);
			sPerm = MLFuncs.sqrt(MLFuncs.columnSum(MLFuncs.square(pBehavLV)));

			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(sPerm.transpose(), refPls.S));
			dp.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(pBehavLV), 
					MLFuncs.abs(refPls.behavLV)));

			progress.endTask(); // curr. permutation
		}

		if(numPermutations != 0) {
			permResult = new PermutationResult();
			permResult.sProb2 = sp.times(1.0 / numPermutations);
			permResult.numPermutations = numPermutations;
			permResult.permSample = reorder;
			permResult.sp = sp;

		}
	}
	//	constructor for fMRI and PET
	public ComputeBehaviorPermutationPls(int imagingType, ComputeBehaviorPls refPls, ConcatenateDatamat st, 
			int numPermutations, StreamedProgressHelper progress, boolean isbehav) throws Exception {

		//		System.out.println("\nPermutation hesaplamasinda\n");

		int numGroups, numCond, k=0;
		Matrix stackedDatamat = null;
		if(imagingType == 2){
			numGroups = st.newDataList.size();
			numCond =  st.num_cond_lst[0];
			stackedDatamat = refPls.stackedDatamat;
		}else{
			numGroups = st.numSubjectList.length; // why it is numSubjectList should it be newDataList????
			numCond = st.numConditions;
			stackedDatamat = st.datamat;
		}

		Matrix sp = new Matrix(refPls.S.getRowDimension(), 1);
		Matrix dp = new Matrix(refPls.behavLV.getRowDimension(), refPls.behavLV.getColumnDimension());


		int[][] reorder = null;
		if(imagingType == 2 && isbehav == false) // if it is PET and Task PLS
		{
			reorder = new RRIPermutationOrder(st.numSubjectList, st.num_cond_lst[0],numPermutations).result;
			//reorder = new int[][]{{1,2},{2, 4},{3,3},{7,5},{8,1},{6,6},{4,8},{5,7},{0,0}};
		}
		if(imagingType != 2 || (imagingType == 2 && isbehav == true)) // if it is fMRI or (PET and 
			// Behavior PLS)
		{
			reorder = new int[stackedDatamat.getRowDimension()][numPermutations];

			for(int p = 0; p < numPermutations; p++) {
				reorder = MLFuncs.setColumn(reorder, p, MLFuncs.randomPermutations(stackedDatamat.
						getRowDimension()));
			}
		}

		System.out.println("r, c"+reorder.length+","+reorder[0].length+" 0 0 is "+reorder[0][0]);

		for(int p = 0; p < numPermutations; p++) {
			progress.startTask("Computing permutation no. " + (p + 1), "Perm. no. " + (p + 1));
			Matrix behavP = null;
			Matrix dataP = null;
			if(imagingType == 2) // it is PET
			{
				if(isbehav == true)
				{
					behavP= MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(reorder, p));
				}
				else
				{   System.out.println("stackedDatamat r,c "+stackedDatamat.getRowDimension()+","
						+stackedDatamat.getColumnDimension());
				dataP =MLFuncs.getRows(stackedDatamat, MLFuncs.getColumn(reorder, p));
				}
			}
			else{ // it is fMRI
				behavP= MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(reorder, p));
			}

			Matrix stackedData = null;

			for(int g = 1; g <= numGroups; g++) {
				if(imagingType == 2) // if it is PET
				{
					k = st.num_cond_lst[g-1];
				}
				int n = st.numSubjectList[g - 1];

				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(st.numSubjectList, 
						MLFuncs.range(0, g - 2))) * numCond;

				System.out.println("p, g, span "+p+","+g+","+span);

				Matrix data = null;

				int[] range = MLFuncs.range(span, n * k + span - 1);

				if(imagingType == 2 && isbehav == false)//task PLS and PET
				{
					if(refPls.singleCondLst == null)
					{
						if(numGroups == 1)
							data = new RRITaskMean(dataP, n).taskMean.minus(new Matrix(k, 1, 1).
									times(MLFuncs.columnMean(dataP)));
						else{
							Matrix tmp =MLFuncs.getRows(dataP, range);
							data = new RRITaskMean(tmp, n).taskMean.minus(new Matrix(k, 1, 1).
									times(MLFuncs.columnMean(tmp)));
						}
					}
					else if (g==0){
						data = new RRITaskMean(dataP, st.numSubjectList).taskMean.minus(
								new Matrix(numGroups, 1, 1).times(MLFuncs.columnMean(dataP)));
					}
				}
				if(imagingType != 2 || (imagingType == 2 && isbehav == true))//do it for fMRI and 
					//PET for behavior
				{
					// Check for upcoming NaN and re-sample if necessary.  This only happened on behavior 
					// analysis, because the 'xcor' inside of 'rri_corr_maps' contains a 'stdev', 
					// which is a divident. If it is 0, it will cause divided by 0 problem. 
					// Since this happend very rarely, so the speed will not be affected that much.
					double min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
					int count = 0;

					while(min1 == 0) {
						reorder = MLFuncs.setColumn(reorder, p, MLFuncs.randomPermutations(stackedDatamat.
								getRowDimension()));
						behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(reorder, p));
						min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
						count++;

						if(count > 100) {
							throw new Exception("Please check your behavior data, and make sure none " +
							"of the columns are all the same for each group");
						}
					}

					//Notice here that stacked_datamat is used, instead of boot_p. This is only for 
					// behavpls_perm.
					if(numGroups == 1) {
						data = new RRICorrMaps(behavP, stackedDatamat, n, k).maps;
					} else {
						data = new RRICorrMaps(MLFuncs.getRows(behavP, range), MLFuncs.getRows(
								stackedDatamat, range), n, k).maps;
					}
				} // end of if

				if(imagingType == 2 && (refPls.singleCondLst == null || g == 0))
					stackedData = MLFuncs.append(stackedData, data);
				else // fMRI
					stackedData = MLFuncs.append(stackedData, data);
			}//end of for numGroups 

			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());

			Matrix sPerm = USV.getS();
			Matrix pBehavLV = USV.getV();

			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.behavLV, pBehavLV).rotatedMatrix;
			pBehavLV = pBehavLV.times(sPerm).times(rotatedMatrix);
			sPerm = MLFuncs.sqrt(MLFuncs.columnSum(MLFuncs.square(pBehavLV)));

			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(sPerm.transpose(), refPls.S));
			dp.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(pBehavLV), MLFuncs.abs(refPls.behavLV)));

			progress.endTask(); // curr. permutation			
		}// end for numpermutation

		if(numPermutations != 0) {
			permResult = new PermutationResult();
			permResult.sProb2 = sp.times(1.0 / numPermutations);
			permResult.numPermutations = numPermutations;
			permResult.permSample = reorder;
			permResult.sp = sp;
					}
	}
}
