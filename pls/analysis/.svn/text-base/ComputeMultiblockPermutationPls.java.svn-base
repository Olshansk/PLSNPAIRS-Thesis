package pls.analysis;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import Jama.Matrix;
import Jama.SingularValueDecomposition;;

public class ComputeMultiblockPermutationPls {

	protected PermutationResult permResult = null;
	//constructor for fMRI
	public ComputeMultiblockPermutationPls(ComputeMultiblockPls refPls, Matrix stackedDatamat, 
			Matrix posthocData, int k, int numPermutations, int[] numSubjectList, int[] bscan, 
			StreamedProgressHelper progress) throws Exception {
		int numGroups = numSubjectList.length;
		int kk = bscan.length;
		
		Matrix sp = new Matrix(refPls.S.getRowDimension(), 1);
		Matrix dp = new Matrix(refPls.V.getRowDimension(), refPls.designLV.getColumnDimension());
		
		int[][] Treorder = new RRIPermutationOrder(numSubjectList, k, numPermutations).result;
		
		int[][] Breorder = new int[stackedDatamat.getRowDimension()][numPermutations];
		
		for(int p = 0; p < numPermutations; p++) {
			Breorder = MLFuncs.setColumn(Breorder, p, MLFuncs.randomPermutations(stackedDatamat.
					getRowDimension()));
		}
		
		for(int p = 0; p < numPermutations; p++) {
			progress.startTask("Computing permutation no. " + (p + 1), "Perm. no. " + (p + 1));
			Matrix dataP = MLFuncs.getRows(stackedDatamat, MLFuncs.getColumn(Treorder, p));
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(Breorder, p));
			
			Matrix stackedTBdata = null;
			Matrix stackedData = null;
			
			for(int g = 1; g <= numGroups; g++) {
				int n = numSubjectList[g - 1];
				
				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, 
						MLFuncs.range(0, g - 2))) * k;
				
				Matrix Tdata = null;

				int[] range = MLFuncs.range(span, n * k + span - 1);
				Matrix tempDataP = MLFuncs.getRows(dataP, range);
				
				if(numGroups == 1) {
					Tdata = new RRITaskMean(dataP, n).taskMean.minus(new Matrix(k, 1, 1).times(MLFuncs.
							columnMean(dataP)));
				} else {
					Tdata = new RRITaskMean(MLFuncs.getRows(tempDataP, range), n).taskMean.minus(
							new Matrix(k, 1, 1).times(MLFuncs.columnMean(tempDataP)));
				}
				
				double min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
				int count = 0;
				
				while(min1 == 0) {
					Breorder = MLFuncs.setColumn(Breorder, p, MLFuncs.randomPermutations(stackedDatamat.
							getRowDimension()));
					behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(Breorder, p));
					min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
					count++;
					
					if(count > 100) {
						// Please check your behavior data, and make sure none of the columns are all the 
						// same for each group
						// Program can not proceed
						throw new Exception();
					}
				}
				
				Matrix Bdata = null;
				
				// Notice here that stacked_datamat is used, instead of boot_p. This is only for 
				// behavpls_perm.
				if(numGroups == 1) {
					Bdata = new RRICorrMapsNotAll(behavP, stackedDatamat, n, bscan).maps;
				} else {
					Bdata = new RRICorrMapsNotAll(MLFuncs.getRows(behavP, range), MLFuncs.getRows(
							stackedDatamat, range), n, bscan).maps;
				}
				
				Matrix TBdata = MLFuncs.append(Tdata, Bdata);
				Matrix data = MLFuncs.append(MLFuncs.normalizeColumn(Tdata), MLFuncs.normalizeColumn(Bdata));

				stackedTBdata = MLFuncs.append(stackedTBdata, TBdata);
				stackedData = MLFuncs.append(stackedData, data);
				progress.endTask(); // curr. permutation	
			}
			
			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());
			
			Matrix sPerm = USV.getS();
			Matrix pV = USV.getV();
			
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.V, pV).rotatedMatrix;
			pV = pV.times(sPerm).times(rotatedMatrix);
			sPerm = MLFuncs.sqrt(MLFuncs.columnSum(MLFuncs.square(pV)));
			
			double pTotalS = MLFuncs.sum(MLFuncs.square(stackedTBdata));
			Matrix sPermTemp = MLFuncs.diag(MLFuncs.square(sPerm));
			Matrix per = sPermTemp.solveTranspose(MLFuncs.columnSum(sPermTemp));
			sPerm = MLFuncs.sqrt(per.times(pTotalS));
			pV = MLFuncs.normalizeRow(pV).times(MLFuncs.diag(sPerm));
			
			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(sPerm, refPls.orgS));
			dp.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(pV), MLFuncs.abs(refPls.orgV)));
			
			Matrix BpV = null;
			for(int g = 1; g <= numGroups; g++) {
				int t = refPls.stackedBehavData.getColumnDimension();
				int start = (g - 1) * k + (g - 1) * kk * t + k;
				int end =   (g - 1) * k + (g - 1) * kk * t + k + kk * t - 1;
				int[] range = MLFuncs.range(start, end);
				BpV = MLFuncs.append(BpV, MLFuncs.getRows(pV, range));
			}
			
			if(posthocData != null) {
				Matrix temp = new RRIXCorr(posthocData, BpV).result;
				refPls.pOrigPost.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(temp), 
						MLFuncs.abs(refPls.origPost)));
			}
		}
		
		if(numPermutations != 0) {
			permResult = new PermutationResult();
			permResult.sProb = sp.times(1.0 / (numPermutations + 1));
			permResult.vProb = dp.times(1.0 / (numPermutations + 1));
			permResult.numPermutations = numPermutations;
			permResult.TpermSamp = Treorder;
			permResult.BPermSamp = Breorder;
			permResult.sp = sp;
			
			if(posthocData != null) {
				permResult.posthocProb = refPls.pOrigPost.times(1.0 / numPermutations);
			}
		}
	}
	
	//constructor for PET and fMRI
	public ComputeMultiblockPermutationPls(int imagingType, ComputeMultiblockPls refPls, 
			ConcatenateDatamat st, int numPermutations, int[] bscan, 
			StreamedProgressHelper progress) throws Exception {
		 
		int numGroups = st.numSubjectList.length;
		int kk = bscan.length;
		int k;
		if(imagingType == 2)
			k= st.num_cond_lst[0];
		else
			k = st.numConditions;		
		
		Matrix sp = new Matrix(refPls.S.getRowDimension(), 1);
		Matrix dp = new Matrix(refPls.V.getRowDimension(), refPls.designLV.getColumnDimension());
		
		int[][] Treorder = new RRIPermutationOrder(st.numSubjectList, k, numPermutations).result;			
		//datamat was stackedDatamat
		int[][] Breorder = new int[st.datamat.getRowDimension()][numPermutations];
		
		for(int p = 0; p < numPermutations; p++) {
			Breorder = MLFuncs.setColumn(Breorder, p, MLFuncs.randomPermutations(st.datamat.
					getRowDimension()));
		}
		
		for(int p = 0; p < numPermutations; p++) {
			progress.startTask("Computing permutation no. " + (p + 1), "Perm. no. " + (p + 1));
			Matrix dataP = MLFuncs.getRows(st.datamat, MLFuncs.getColumn(Treorder, p));
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(Breorder, p));
			
			Matrix stackedTBdata = null;
			Matrix stackedData = null;
			
			for(int g = 1; g <= numGroups; g++) {
				int n = st.numSubjectList[g - 1];
				int span;
				if(imagingType==2){
					k = st.num_cond_lst[g-1];
					span = MLFuncs.sum(MLFuncs.getItemsAtIndices(st.numSubjectList, 
							MLFuncs.range(0, g - 2))) * st.num_cond_lst[0];
				}
				else
					span = MLFuncs.sum(MLFuncs.getItemsAtIndices(st.numSubjectList, 
							MLFuncs.range(0, g - 2))) * k;
				
				Matrix Tdata = null;

				int[] range = MLFuncs.range(span, n * k + span - 1);
				Matrix tempDataP = MLFuncs.getRows(dataP, range);
				
				if(numGroups == 1) {
					Tdata = new RRITaskMean(dataP, n).taskMean.minus(new Matrix(k, 1, 1).times(MLFuncs.
							columnMean(dataP)));
				} else {
					Tdata = new RRITaskMean(MLFuncs.getRows(tempDataP, range), n).taskMean.minus(new 
							Matrix(k, 1, 1).times(MLFuncs.columnMean(tempDataP)));
				}
				
				double min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
				int count = 0;
				
				while(min1 == 0) {
					Breorder = MLFuncs.setColumn(Breorder, p, MLFuncs.randomPermutations(st.datamat.
							getRowDimension()));
					behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(Breorder, p));
					min1 = MLFuncs.min(MLFuncs.std(MLFuncs.getRows(behavP, range)));
					count++;
					
					if(count > 100) {
						// Please check your behavior data, and make sure none of the columns are all the 
						// same for each group
						// Program can not proceed
						throw new Exception();
					}
				}
				
				Matrix Bdata = null;
				
				// Notice here that stacked_datamat is used, instead of boot_p. This is only for 
				// behavpls_perm.
				if(numGroups == 1) {
					Bdata = new RRICorrMapsNotAll(behavP, st.datamat, n, bscan).maps;
				} else {
					Bdata = new RRICorrMapsNotAll(MLFuncs.getRows(behavP, range), MLFuncs.getRows(
							st.datamat, range), n, bscan).maps;
				}
				
				Matrix TBdata = MLFuncs.append(Tdata, Bdata);
				Matrix data = MLFuncs.append(MLFuncs.normalizeColumn(Tdata), MLFuncs.normalizeColumn(Bdata));

				stackedTBdata = MLFuncs.append(stackedTBdata, TBdata);
				stackedData = MLFuncs.append(stackedData, data);
				progress.endTask(); // curr. permutation
			}
			
			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());
			
			Matrix sPerm = USV.getS();
			Matrix pV = USV.getV();
			
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.V, pV).rotatedMatrix;
			pV = pV.times(sPerm).times(rotatedMatrix);
			sPerm = MLFuncs.sqrt(MLFuncs.columnSum(MLFuncs.square(pV)));
			
			double pTotalS = MLFuncs.sum(MLFuncs.square(stackedTBdata));
			Matrix sPermTemp = MLFuncs.diag(MLFuncs.square(sPerm));
			Matrix per = sPermTemp.solveTranspose(MLFuncs.columnSum(sPermTemp));
			sPerm = MLFuncs.sqrt(per.times(pTotalS));
			pV = MLFuncs.normalizeRow(pV).times(MLFuncs.diag(sPerm));
			
			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(sPerm, refPls.orgS));
			dp.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(pV), MLFuncs.abs(refPls.orgV)));
			
			Matrix BpV = null;
			for(int g = 1; g <= numGroups; g++) {
				int t = refPls.stackedBehavData.getColumnDimension();
				int start = (g - 1) * k + (g - 1) * kk * t + k;
				int end =   (g - 1) * k + (g - 1) * kk * t + k + kk * t - 1;
				int[] range = MLFuncs.range(start, end);
				BpV = MLFuncs.append(BpV, MLFuncs.getRows(pV, range));
			}
			
			if(st.posthocData != null) {
				Matrix temp = new RRIXCorr(st.posthocData, BpV).result;
				refPls.pOrigPost.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(temp), 
						MLFuncs.abs(refPls.origPost)));
			}
		}
		
		if(numPermutations != 0) {
			permResult = new PermutationResult();
			permResult.sProb = sp.times(1.0 / (numPermutations + 1));
			permResult.vProb = dp.times(1.0 / (numPermutations + 1));
			permResult.numPermutations = numPermutations;
			permResult.TpermSamp = Treorder;
			permResult.BPermSamp = Breorder;
			permResult.sp = sp;
			
			if(st.posthocData != null) {
				permResult.posthocProb = refPls.pOrigPost.times(1.0 / numPermutations);
			}
		}
	}
	
}
