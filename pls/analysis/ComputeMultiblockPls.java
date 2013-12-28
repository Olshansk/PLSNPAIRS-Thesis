package pls.analysis;

import java.util.Vector;

import pls.shared.MLFuncs;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputeMultiblockPls {
	
	protected Vector<Matrix> datamatCorrsList = new Vector<Matrix>();
	
	protected Vector<Matrix> behavDataList = new Vector<Matrix>();
	
	protected Matrix brainLV = null;
	
	protected Matrix S = null;
	
	protected Matrix V = null;
	
	protected Matrix designLV = null;
	
	protected Matrix behavLV = null;
	
	protected Matrix brainScores = null;
	
	protected Matrix designScores = null;
	
	protected Matrix behavScores = null;
	
	protected Matrix bScores = null;
	
	protected Matrix lvCorrs = null;
	
	protected Matrix orgS = null;
	
	protected Matrix orgV = null;
	
	protected Matrix origPost = null;
	
	protected Matrix pOrigPost = null;
	
	protected int[] behavRowIdx = null;
	
	protected Matrix stackedBehavData = null;
	
	protected int[] rowIdx = null;
	//constructor for fMRI
public ComputeMultiblockPls(Matrix stackedDatamat, Matrix behavData, Matrix posthocData, Vector<Matrix> behavDataList, Vector<Matrix> newDataList, int[] numSubjectList, int numConditions, int[] bscan) {
		
		Matrix stackedTBdatamatCorrs = null;
		Matrix stackedDatamatCorrs = null;
		stackedBehavData = behavData.copy();
		this.behavDataList = behavDataList;
		
		int numGroups = newDataList.size();
		
		int k = numConditions;
		int kk = bscan.length;
		
		// loop accross the groups, and calculate datamatcorrs for each group
		for(int i = 0; i < numGroups; i++) {
			int n = numSubjectList[i];
			
			Matrix datamat = newDataList.get(i);
			
			// Compute task mean
			Matrix TdatamatCorrs = new RRITaskMean(datamat, n).taskMean.minus(new Matrix(k, 1, 1).times(MLFuncs.columnMean(datamat)));
			
			// Compute correlation
			Matrix BdatamatCorrs = new RRICorrMapsNotAll(behavDataList.get(i), datamat, n, bscan).maps;
			
			// Stack task and behavior - keep un-normalize data that will be used to recover the normalized one
			Matrix TBdatamatCorrs = MLFuncs.append(TdatamatCorrs, BdatamatCorrs);
			
			// Stack task and behavior - normalize to unit length to reduce scaling difference
			Matrix datamatCorrs = MLFuncs.append(MLFuncs.normalizeColumn(TdatamatCorrs), MLFuncs.normalizeColumn(BdatamatCorrs));
			
			stackedTBdatamatCorrs = MLFuncs.append(stackedTBdatamatCorrs, TBdatamatCorrs);
			stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
			datamatCorrsList.add(BdatamatCorrs);
			
			int[] range = MLFuncs.range(0, stackedBehavData.getColumnDimension() * k - 1);
			int[][] tempRowIdx = MLFuncs.reshape(new int[][]{range}, stackedBehavData.getColumnDimension(), k);
			tempRowIdx = MLFuncs.getColumns(tempRowIdx, bscan);
			
			rowIdx = MLFuncs.append(rowIdx, MLFuncs.plus(MLFuncs.flattenHorizontally(tempRowIdx), stackedBehavData.getColumnDimension() * k * i));
		}
		
		if(posthocData != null) {
			posthocData = MLFuncs.getRows(posthocData, rowIdx);
		}
		
		SingularValueDecomposition USV = new SingularValueDecomposition(stackedDatamatCorrs.transpose());
		
		brainLV = USV.getU();
		S = MLFuncs.diag(USV.getS()).transpose();
		V = USV.getV();
		
		/** Matrix originalV = V.times(USV.getS()); // Needed?? **/
		
		// Since the 2 matrices that went into the SVD were unit normal, we should
		// go backwards from the total Singular value Sum of Squares (SSQ)
		
		// Calculate total SSQ
		double totalS = MLFuncs.sum(MLFuncs.square(stackedTBdatamatCorrs).getRowPackedCopy());
		
		// Calculate distribution of normalized SSQ across LVs
		Matrix sSq = MLFuncs.square(S);
		Matrix per = sSq.times(1.0 / MLFuncs.sum(sSq));
		
		// Re-calculate singular value based on the distribution of SSQ across normalized LVs
		orgS = MLFuncs.sqrt(per.times(totalS));
		
		// Re-scale v (block LV) with singular value
		orgV = V.times(MLFuncs.diag(orgS).transpose());
		
		// Separate v into 2 parts: designlv and behavlv
		for(int g = 1; g <= numGroups; g++) {
			int t = stackedBehavData.getColumnDimension();
			
			int start = (g - 1) * k + (g - 1) * kk * t;
			int end =   (g - 1) * k + (g - 1) * kk * t + k - 1;
			int[] range = MLFuncs.range(start, end);
			designLV = MLFuncs.append(designLV, MLFuncs.getRows(V, range));
			
			start = (g - 1) * k + (g - 1) * kk * t + k;
			end =   (g - 1) * k + (g - 1) * kk * t + k + kk * t - 1;
			range = MLFuncs.range(start, end);
			behavLV = MLFuncs.append(behavLV, MLFuncs.getRows(V, range));
		}
		
		int numColumns = designLV.getColumnDimension();
		
		// Expand the num_subj for each row (cond)
		rowIdx = null;
		int last = 0;
		
		for(int g = 1; g <= numGroups; g++) {
			int n = numSubjectList[g - 1];
			
			int start = (g - 1) * k;
			int end =   (g - 1) * k + k - 1;
			int[] range = MLFuncs.range(start, end);
			Matrix temp = MLFuncs.reshape(MLFuncs.getRows(designLV, range), 1, numColumns * k);
			temp = MLFuncs.replicateRows(temp, n);
			temp = MLFuncs.reshape(temp, n * k, numColumns);
			
			designScores = MLFuncs.append(designScores, temp);
			
			// Take this advantage (having g & n) to get row_idx
			int[][] temp2 = new int[][]{ MLFuncs.range(0, n * k - 1) };
			temp2 = MLFuncs.reshape(temp2, n, k);
			int[] temp3 = MLFuncs.flattenHorizontally(MLFuncs.transpose(MLFuncs.getColumns(temp2, bscan)));
			
			behavDataList.set(g - 1, MLFuncs.getRows(behavDataList.get(g - 1), temp3));
			
			rowIdx = MLFuncs.append(rowIdx, MLFuncs.plus(temp3, last));
			last += n * k;
		}
		
		behavRowIdx = rowIdx.clone();
		behavData = MLFuncs.getRows(behavData, rowIdx);
		
		// Calculate behav scores
		bScores = stackedDatamat.times(brainLV);
		RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(MLFuncs.getRows(stackedDatamat, rowIdx), MLFuncs.getRows(stackedBehavData, rowIdx), brainLV, behavLV, kk, numSubjectList);
		brainScores = rgb.scores;
		behavScores = rgb.fScores;
		lvCorrs = rgb.lvCorrs;
		
		if(posthocData != null) {
			origPost = new RRIXCorr(posthocData, behavLV).result;
			pOrigPost = new Matrix(origPost.getRowDimension(), origPost.getColumnDimension());
		}
	}
	//this constructor works for fMRI and PET
	public ComputeMultiblockPls(int imagingType, ConcatenateDatamat st, int[] bscan) {
		Matrix stackedTBdatamatCorrs = null;
		Matrix stackedDatamatCorrs = null;
		
		if(imagingType !=2) // done if it is not PET
			stackedBehavData = st.behavData.copy();
	
	
		this.behavDataList = st.behavDataList;
		
		int numGroups = st.newDataList.size();
		int k;
		if (imagingType == 2)//if PET
			k = st.num_cond_lst[0];
		else
			k = st.numConditions;
	
		int kk = bscan.length;
		
		// loop accross the groups, and calculate datamatcorrs for each group
		for(int i = 0; i < numGroups; i++) {
			int n = st.numSubjectList[i];
			
			Matrix datamat = st.newDataList.get(i);
			
			// Compute task mean
			Matrix TdatamatCorrs = new RRITaskMean(datamat, n).taskMean.minus(new Matrix(k, 1, 1).times(MLFuncs.columnMean(datamat)));
			
			// Compute correlation
			Matrix BdatamatCorrs = new RRICorrMapsNotAll(behavDataList.get(i), datamat, n, bscan).maps;
			
			// Stack task and behavior - keep un-normalize data that will be used to recover the normalized one
			Matrix TBdatamatCorrs = MLFuncs.append(TdatamatCorrs, BdatamatCorrs);
			
			// Stack task and behavior - normalize to unit length to reduce scaling difference
			Matrix datamatCorrs = MLFuncs.append(MLFuncs.normalizeColumn(TdatamatCorrs), MLFuncs.normalizeColumn(BdatamatCorrs));
			
			if(imagingType ==2) // done if it is PET
				stackedBehavData = MLFuncs.append(stackedBehavData, st.behavDataList.get(i));
		
			stackedTBdatamatCorrs = MLFuncs.append(stackedTBdatamatCorrs, TBdatamatCorrs);
			stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
			datamatCorrsList.add(BdatamatCorrs);
			
			int[] range = MLFuncs.range(0, stackedBehavData.getColumnDimension() * k - 1);
			int[][] tempRowIdx = MLFuncs.reshape(new int[][]{range}, stackedBehavData.getColumnDimension(), k);
			tempRowIdx = MLFuncs.getColumns(tempRowIdx, bscan);
			
			rowIdx = MLFuncs.append(rowIdx, MLFuncs.plus(MLFuncs.flattenHorizontally(tempRowIdx), stackedBehavData.getColumnDimension() * k * i));
		}
		
		if(st.posthocData != null) {
			st.posthocData = MLFuncs.getRows(st.posthocData, rowIdx);
		}
		
		SingularValueDecomposition USV = new SingularValueDecomposition(stackedDatamatCorrs.transpose());
		
		brainLV = USV.getU();
		S = MLFuncs.diag(USV.getS()).transpose();
		V = USV.getV();
		
		/** Matrix originalV = V.times(USV.getS()); // Needed?? **/ 
		
		// Since the 2 matrices that went into the SVD were unit normal, we should
		// go backwards from the total Singular value Sum of Squares (SSQ)
		
		// Calculate total SSQ
		double totalS = MLFuncs.sum(MLFuncs.square(stackedTBdatamatCorrs).getRowPackedCopy());
		
		// Calculate distribution of normalized SSQ across LVs
		Matrix sSq = MLFuncs.square(S);
		Matrix per = sSq.times(1.0 / MLFuncs.sum(sSq));
		
		// Re-calculate singular value based on the distribution of SSQ across normalized LVs
		orgS = MLFuncs.sqrt(per.times(totalS));
		
		// Re-scale v (block LV) with singular value
		orgV = V.times(MLFuncs.diag(orgS).transpose());
		
		// Separate v into 2 parts: designlv and behavlv
		for(int g = 1; g <= numGroups; g++) {
			int t = stackedBehavData.getColumnDimension();
			
			int start = (g - 1) * k + (g - 1) * kk * t;
			int end =   (g - 1) * k + (g - 1) * kk * t + k - 1;
			int[] range = MLFuncs.range(start, end);
			designLV = MLFuncs.append(designLV, MLFuncs.getRows(V, range));
			
			start = (g - 1) * k + (g - 1) * kk * t + k;
			end =   (g - 1) * k + (g - 1) * kk * t + k + kk * t - 1;
			range = MLFuncs.range(start, end);
			behavLV = MLFuncs.append(behavLV, MLFuncs.getRows(V, range));
		}
		
		int numColumns = designLV.getColumnDimension();
		
		// Expand the num_subj for each row (cond)
		rowIdx = null;
		int last = 0;
		
		for(int g = 1; g <= numGroups; g++) {
			int n = st.numSubjectList[g - 1];
			
			int start = (g - 1) * k;
			int end =   (g - 1) * k + k - 1;
			int[] range = MLFuncs.range(start, end);
			Matrix temp = MLFuncs.reshape(MLFuncs.getRows(designLV, range), 1, numColumns * k);
			temp = MLFuncs.replicateRows(temp, n);
			temp = MLFuncs.reshape(temp, n * k, numColumns);
			
			designScores = MLFuncs.append(designScores, temp);
			
			// Take this advantage (having g & n) to get row_idx
			int[][] temp2 = new int[][]{ MLFuncs.range(0, n * k - 1) };
			temp2 = MLFuncs.reshape(temp2, n, k);
			int[] temp3 = MLFuncs.flattenHorizontally(MLFuncs.transpose(MLFuncs.getColumns(temp2, bscan)));
			
			behavDataList.set(g - 1, MLFuncs.getRows(behavDataList.get(g - 1), temp3));
			
			rowIdx = MLFuncs.append(rowIdx, MLFuncs.plus(temp3, last));
			last += n * k;
		}
		
		behavRowIdx = rowIdx.clone();
		if(imagingType !=2) // done if it is not PET
			st.behavData = MLFuncs.getRows(st.behavData, rowIdx);// not included in PET
	
		// Calculate behav scores
		bScores = st.datamat.times(brainLV);
		RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(MLFuncs.getRows(st.datamat, rowIdx), MLFuncs.getRows(stackedBehavData, rowIdx), brainLV, behavLV, kk, st.numSubjectList);
		brainScores = rgb.scores;
		behavScores = rgb.fScores;
		lvCorrs = rgb.lvCorrs;
		
		if(st.posthocData != null) {
			origPost = new RRIXCorr(st.posthocData, behavLV).result;
			pOrigPost = new Matrix(origPost.getRowDimension(), origPost.getColumnDimension());
		}
	}
}
