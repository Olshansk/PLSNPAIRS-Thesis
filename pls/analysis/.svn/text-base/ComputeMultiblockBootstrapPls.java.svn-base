package pls.analysis;

import java.util.Vector;

import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.stat.descriptive.rank.Percentile;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputeMultiblockBootstrapPls {

	protected BootstrapResult bootResult = null;
	
	private final int maxSubjectsPerGroup = 8;
	//constructor for fMRI
	public ComputeMultiblockBootstrapPls(ComputeMultiblockPls refPls, Matrix stackedDatamat, 
			int numConditions, int[] eventList, int numBootstraps, int[] numSubjectList, 
			int minSubjectsPerGroup, boolean[] isBootstrapSamples, Vector<int[][]> bootstrapSamples, 
			int newNumBootstraps, int[] bscan, double confidenceLimit, StreamedProgressHelper progress) 
			throws Exception {
		
		if(numBootstraps == 0) {
			return;
		}
		
		Matrix origCorr = refPls.lvCorrs;
		int r1 = origCorr.getRowDimension();
		int c1 = origCorr.getColumnDimension();
		
		int numGroups = numSubjectList.length;
		
		// Keeps track of number of times a new bootstrap had to be generated
		int countNewTotal = 0;
		Vector<Matrix> badBeh = new Vector<Matrix>();
		
		RRIBootstrapOrder rbo = new RRIBootstrapOrder(numSubjectList, numConditions, numBootstraps, false, 
				minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0);
		
		int[][] bootOrder = rbo.result;
		/*int[][] bootOrder ={{0, 0, 0},
				{2, 0, 1},
				{2, 1, 2},
				{3, 3, 3},
				{5, 3, 4},
				{5, 4, 5},
				{6, 6, 6},
				{8, 6, 7},
				{8, 7, 8},
				{9, 9, 9},
				{11, 9, 10},
				{11, 10, 11},
				{12, 12, 12},
				{14, 12, 13},
				{14, 13, 14},
				{15, 15, 15},
				{17, 15, 16},
				{17, 16, 17},
				{18, 18, 18},
				{20, 18, 19},
				{20, 19, 20}};*/
		
		if(rbo.newNumBootstraps != numBootstraps) {
			numBootstraps = rbo.newNumBootstraps;
			
			// TODO: Something else?? exception?  Allow user to choose new num boot?
		}
		
		Matrix[] distribution = new Matrix[numBootstraps + 1];
		
		distribution[0] = origCorr;
		
		if(MLFuncs.sum(MLFuncs.lessThanOrEqualTo(numSubjectList, maxSubjectsPerGroup)) == numGroups) {
			isBootstrapSamples = new boolean[numGroups];
			for(int i = 0; i < isBootstrapSamples.length; i++) {
				isBootstrapSamples[i] = true;
			}
		} else {
			for(int i = 0; i < isBootstrapSamples.length; i++) {
				isBootstrapSamples[i] = false;
			}
		}
		
		if(bootOrder.length == 0) {
			return;
		}

		Matrix diagS = MLFuncs.diag(refPls.S);
		Matrix originalSal = refPls.brainLV.times(diagS);
		Matrix originalBehavLV = refPls.V.times(diagS);
		
		Matrix salienceSum = originalSal.copy();
		Matrix salienceSq = MLFuncs.square(salienceSum);
		Matrix designSalienceSum = originalBehavLV.copy();
		Matrix designSalienceSq = MLFuncs.square(designSalienceSum);
		
		int k = numConditions;
		int kk = bscan.length;
		
		for(int p = 0; p < numBootstraps; p++) {
			progress.startTask("Computing bootstrap no. " + (p + 1), "Bootstrap no. " + (p + 1));
			Matrix dataP = MLFuncs.getRows(stackedDatamat, MLFuncs.getColumn(bootOrder, p));
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
			
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
				
				for(boolean b : isBootstrapSamples) {
					if(!b) {
						// The code below is mainly trying to find a proper reorder matrix
						
						// Init badbehav cell array to 0 which is used to record the bad behav data caused 
						// by bad re-order. This var. is for disp only.
						Matrix badBehav = new Matrix(numConditions, refPls.behavDataList.get(g - 1).
								getColumnDimension());
						
						// Check for upcoming NaN and re-sample if necessary. This only happened on 
						// behavior analysis, because the 'xcor' inside of 'corr_maps' contains a 'stdev', 
						// which is a divident. If it is 0, it will cause divided by 0 problem. 
						// Since this happend very rarely, so the speed will not be affected that much.
						
						// For behavpls_boot, also need to account for multiple scans and behavs
						Matrix stdMat = new Matrix(k, refPls.stackedBehavData.getColumnDimension());
						for(int c = 1; c <= k; c++) {
							int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span);
							int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), range2);
							stdMat.setMatrix(c, c, 0, refPls.stackedBehavData.getColumnDimension() - 1, 
									MLFuncs.std(MLFuncs.getRows(refPls.stackedBehavData, range3)));
						}
						
						while(MLFuncs.containsZero(stdMat)) {
							// Keep track of scan & behav that force a resample
							double[] flatStdMat = stdMat.getRowPackedCopy();
							for(int i = 0; i < flatStdMat.length; i++) {
								if(flatStdMat[i] == 0) {
									badBehav.set(i, 0, badBehav.get(i, 0) + 1);
								}
							}
							
							badBeh.add(badBehav);
							
							if(countNewTotal > numBootstraps) {
								// Please check behavior data
								throw new Exception();
							}
							
							int[][] reorderP = new RRIBootstrapOrder(numSubjectList, numConditions, 
									numBootstraps, true, minSubjectsPerGroup, isBootstrapSamples, 
									bootstrapSamples, newNumBootstraps, 50.0).result;
							MLFuncs.setColumn(bootOrder, p, MLFuncs.getColumn(reorderP, p));
							
							// Recalculate stdMat
							for(int c = 1; c <= k; c++) {
								int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span);
								int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), 
										range2);
								stdMat.setMatrix(c, c, 0, refPls.stackedBehavData.getColumnDimension() - 1, 
										MLFuncs.std(MLFuncs.getRows(refPls.stackedBehavData, range3)));
							}
						}
						break;
					}
				}
				
				// Now, we can use this proper reorder matrix to generate behav_p & data_p, and then to
				// calculate datamatcoors
				behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
				dataP = MLFuncs.getRows(stackedDatamat, MLFuncs.getColumn(bootOrder, p));

				Matrix Bdata = null;
				
				if(numGroups == 1) {
					Bdata = new RRICorrMapsNotAll(behavP, dataP, n, bscan).maps;
				} else {
					Bdata = new RRICorrMapsNotAll(MLFuncs.getRows(behavP, range), MLFuncs.getRows(dataP, 
							range), n, bscan).maps;
				}
				
				// Matrix TBdata = MLFuncs.append(Tdata, Bdata); needed?
				Matrix data = MLFuncs.append(MLFuncs.normalizeColumn(Tdata), MLFuncs.normalizeColumn(Bdata));
				
				stackedData = MLFuncs.append(stackedData, data);
				progress.endTask(); // curr. bootstrap
			}
			
			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());
			
			Matrix pBrainLV = USV.getU();
			Matrix sBoot = USV.getS();
			Matrix pBehavLV = USV.getV();
			
			// Rotate pbehavlv to align with the original behavlv
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.V, pBehavLV).rotatedMatrix;
			
			Matrix temp = sBoot.times(rotatedMatrix);
			pBrainLV = pBrainLV.times(temp);
			pBehavLV = pBehavLV.times(temp);
			
			RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(MLFuncs.getRows(dataP, refPls.rowIdx), 
					MLFuncs.getRows(behavP, refPls.rowIdx), MLFuncs.normalizeRow(pBrainLV), 
					MLFuncs.normalizeRow(pBehavLV), kk, numSubjectList);
			Matrix bCorr = rgb.lvCorrs;
			
			distribution[p + 1] = bCorr;

			salienceSum.plusEquals(pBrainLV);
			salienceSq.plusEquals(MLFuncs.square(pBrainLV));
			designSalienceSum.plusEquals(pBehavLV);
			designSalienceSq.plusEquals(MLFuncs.square(pBehavLV));
		}

		Matrix salienceSum2 = MLFuncs.square(salienceSum).times(1.0 / (numBootstraps + 1));
		Matrix designSalienceSum2 = MLFuncs.square(designSalienceSum).times(1.0 / (numBootstraps + 1));
		
		// Compute standard errors - standard deviation of bootstrap sample since original sample is part of
		// bootstrap, divide by number of bootstrap iterations rather than number of bootstraps minus 1
		
		// Add ceiling to calculations to prevent the following operations from producing 
		// negative/complex numbers
		Matrix brainStandardErrors = MLFuncs.sqrt(salienceSq.minus(salienceSum2).times(1.0 / numBootstraps));
		Matrix behavStandardErrors = MLFuncs.sqrt(designSalienceSq.minus(designSalienceSum2).
				times(1.0 / numBootstraps));
		
		int[] brainZeros = MLFuncs.findLessThanOrEqualTo(brainStandardErrors, 0);
		brainStandardErrors = MLFuncs.setValues(brainStandardErrors, brainZeros, 1);
		
		int[] behavZeros = MLFuncs.findLessThanOrEqualTo(behavStandardErrors, 0);
		behavStandardErrors = MLFuncs.setValues(behavStandardErrors, behavZeros, 1);
		
		Matrix compare = originalSal.arrayRightDivide(brainStandardErrors);
		Matrix compareBehavLV = originalBehavLV.arrayRightDivide(behavStandardErrors);
		
		compare = MLFuncs.setValues(compare, brainZeros, 0);
		compareBehavLV = MLFuncs.setValues(compareBehavLV, behavZeros, 0);
		
		double ul = confidenceLimit;
		double ll = 100.0 - confidenceLimit;
		
		// e.g. 0.05 >> 0.025 for upper & lower tails, two-tailed
		double cLimNi = 0.5 * (1 - (confidenceLimit * 0.01));

		Matrix ulCorr = new Matrix(r1, c1);
		Matrix llCorr = new Matrix(r1, c1);
		Matrix prop = new Matrix(r1, c1);
		Matrix ulCorrAdj = new Matrix(r1, c1);
		Matrix llCorrAdj = new Matrix(r1, c1);
		// Loop to calculate upper and lower CI limits
		for(int r = 0; r < r1; r++) {
			for(int c = 0; c < c1; c++) {
				double distrib[] = new double[numBootstraps];
				for(int i = 1; i < numBootstraps + 1; i++) {
					distrib[i - 1] = distribution[i].get(r, c);
				}
				
				Percentile per = new Percentile();
				ulCorr.set(r, c, per.evaluate(distrib, ul));
				llCorr.set(r, c, per.evaluate(distrib, ll));
				prop.set(r, c, ((double)MLFuncs.findLessThanOrEqualTo(distrib, 
						origCorr.get(r, c)).length) / numBootstraps);
				
				if(prop.get(r, c) == 1 || prop.get(r, c) == 0) {
					// Can't calculate the cumulative_gaussian_inv
					llCorrAdj.set(r, c, Double.NaN);
					ulCorrAdj.set(r, c, Double.NaN);
				} else {
					// Adjusted confidence intervals - in case the bootstrap samples are extremely skewed
					
					// Norm inverse to start to adjust conf int
					NormalDistributionImpl normalDistribution = new NormalDistributionImpl();
					double ni = normalDistribution.inverseCumulativeProbability(prop.get(r, c));
					
					// 1st part of recalc the lower conf interval, this evaluates to +1.96 for 95%CI
					double uli = 2 * ni + normalDistribution.inverseCumulativeProbability(1 - cLimNi);
					
					// 1st part of recalc the upper conf interval e.g -1.96 for 95%CI
					double lli = 2 * ni + normalDistribution.inverseCumulativeProbability(cLimNi);
					
					// percentile for lower bounds
					double nCDF_lli =  normalDistribution.cumulativeProbability(lli) * 100;
					
					// percentile for upper bounds
					double nCDF_uli =  normalDistribution.cumulativeProbability(uli) * 100;
					
					// new percentile
					llCorrAdj.set(r, c, per.evaluate(distrib, nCDF_lli));
					ulCorrAdj.set(r, c, per.evaluate(distrib, nCDF_uli));
				}
			}
		}
		
		bootResult = new BootstrapResult();
		bootResult.numBootstraps = numBootstraps;
		bootResult.origCorr = origCorr;
		bootResult.ulCorr = ulCorr;
		bootResult.llCorr = llCorr;
		bootResult.ulCorrAdj = ulCorrAdj;
		bootResult.llCorrAdj = llCorrAdj;
		bootResult.prop = prop;
		bootResult.distrib = distribution;
		bootResult.origBrainLV = originalSal;
		bootResult.brainStandardErrors = brainStandardErrors;
		bootResult.compare = compare;
		bootResult.compareBehavLV = compareBehavLV;
		bootResult.bootSample = bootOrder;
		bootResult.badBeh = badBeh;
		bootResult.countNewTotal = countNewTotal;
	}
	//Constructor for fMRI and PET
	public ComputeMultiblockBootstrapPls(int imagingType, ComputeMultiblockPls refPls, ConcatenateDatamat st, 
			int numBootstraps, int minSubjectsPerGroup, boolean[] isBootstrapSamples, 
			Vector<int[][]> bootstrapSamples, int newNumBootstraps, int[] bscan, double confidenceLimit, 
			StreamedProgressHelper progress) throws Exception {
		
		if(numBootstraps == 0) {
			return;
		}
		
		Matrix origCorr = refPls.lvCorrs;
		int r1 = origCorr.getRowDimension();
		int c1 = origCorr.getColumnDimension();
		
		int numGroups;
		if(imagingType == 2)
			numGroups = st.newDataList.size(); 
		else
			numGroups = st.numSubjectList.length;
		
		// Keeps track of number of times a new bootstrap had to be generated
		int countNewTotal = 0;
		Vector<Matrix> badBeh = new Vector<Matrix>();
		
		RRIBootstrapOrder rbo;
		if(imagingType == 2)
			rbo = new RRIBootstrapOrder(st.numSubjectList, st.num_cond_lst[0], numBootstraps, false, 
					minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0);
		else
			rbo = new RRIBootstrapOrder(st.numSubjectList, st.numConditions, numBootstraps, false, 
					minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0);
		
		int[][] bootOrder = rbo.result;
		/*int[][] bootOrder ={{0, 0, 0},
				{2, 0, 1},
				{2, 1, 2},
				{3, 3, 3},
				{5, 3, 4},
				{5, 4, 5},
				{6, 6, 6},
				{8, 6, 7},
				{8, 7, 8},
				{9, 9, 9},
				{11, 9, 10},
				{11, 10, 11},
				{12, 12, 12},
				{14, 12, 13},
				{14, 13, 14},
				{15, 15, 15},
				{17, 15, 16},
				{17, 16, 17},
				{18, 18, 18},
				{20, 18, 19},
				{20, 19, 20}};*/
		
		if(rbo.newNumBootstraps != numBootstraps) {
			numBootstraps = rbo.newNumBootstraps;
			
			// TODO: Something else?? exception?  Allow user to choose new num boot?
		}
		
		Matrix[] distribution = new Matrix[numBootstraps + 1];
		
		distribution[0] = origCorr;
		
		if(MLFuncs.sum(MLFuncs.lessThanOrEqualTo(st.numSubjectList, maxSubjectsPerGroup)) == numGroups) {
			isBootstrapSamples = new boolean[numGroups];
			for(int i = 0; i < isBootstrapSamples.length; i++) {
				isBootstrapSamples[i] = true;
			}
		} else {
			for(int i = 0; i < isBootstrapSamples.length; i++) {
				isBootstrapSamples[i] = false;
			}
		}
		
		if(bootOrder.length == 0) {
			return;
		}

		Matrix diagS = MLFuncs.diag(refPls.S);
		Matrix originalSal = refPls.brainLV.times(diagS);
		Matrix originalBehavLV = refPls.V.times(diagS);
		
		Matrix salienceSum = originalSal.copy();
		Matrix salienceSq = MLFuncs.square(salienceSum);
		Matrix designSalienceSum = originalBehavLV.copy();
		Matrix designSalienceSq = MLFuncs.square(designSalienceSum);
		
		int k;
		int num_cond;
		
		if(imagingType == 2){
			k = st.num_cond_lst[0];
			num_cond = st.num_cond_lst[0];
		}
		else{
			k = st.numConditions;
			num_cond = st.numConditions; 
		}
		int kk = bscan.length;
		
		
		for(int p = 0; p < numBootstraps; p++) {
			progress.startTask("Computing bootstrap no. " + (p + 1), "Bootstrap no. " + (p + 1));
			//st.datamat contains stackedDatamat
			Matrix dataP = MLFuncs.getRows(st.datamat, MLFuncs.getColumn(bootOrder, p));
			Matrix behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
			
			Matrix stackedData = null;
			
			for(int g = 1; g <= numGroups; g++) {
				
				int n = st.numSubjectList[g - 1];
				
				if(imagingType == 2){
					k = st.num_cond_lst[g-1];
									
				//int span = MLFuncs.sum(MLFuncs.getItemsAtIndeces(st.numSubjectList, 
			    // 	MLFuncs.range(0, g - 2))) * k;
					
				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(st.numSubjectList, 
						MLFuncs.range(0, g - 2))) * num_cond;
					
				Matrix Tdata = null;

				int[] range = MLFuncs.range(span, n * k + span - 1);
				Matrix tempDataP = MLFuncs.getRows(dataP, range);
				
				if(numGroups == 1) {
					Tdata = new RRITaskMean(dataP, n).taskMean.minus(new Matrix(k, 1, 1).times(
							MLFuncs.columnMean(dataP)));
				} else {
					Tdata = new RRITaskMean(MLFuncs.getRows(tempDataP, range), n).taskMean.minus(
							new Matrix(k, 1, 1).times(MLFuncs.columnMean(tempDataP)));
				}
				
				for(boolean b : isBootstrapSamples) {
					if(!b) {
						// The code below is mainly trying to find a proper reorder matrix
						
						// Init badbehav cell array to 0 which is used to record the bad behav data 
						// caused by bad re-order. This var. is for disp only.
						Matrix badBehav = new Matrix(num_cond, refPls.behavDataList.get(g - 1).
								getColumnDimension());
						
						// Check for upcoming NaN and re-sample if necessary. This only happened on 
						// behavior analysis, because the 'xcor' inside of 'corr_maps' contains a 'stdev', 
						// which is a divident. If it is 0, it will cause divided by 0 problem. 
						// Since this happend very rarely, so the speed will not be affected that much.
						
						// For behavpls_boot, also need to account for multiple scans and behavs
						Matrix stdMat = new Matrix(k, refPls.stackedBehavData.getColumnDimension());
						for(int c = 1; c <= k; c++) {
							int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span);
							int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), range2);
							stdMat.setMatrix(c, c, 0, refPls.stackedBehavData.getColumnDimension() - 1, 
									MLFuncs.std(MLFuncs.getRows(refPls.stackedBehavData, range3)));
						}
						
						while(MLFuncs.containsZero(stdMat)) {
							if(imagingType == 2)
								countNewTotal++;
							//Keep track of scan & behav that force a resample
							double[] flatStdMat = stdMat.getRowPackedCopy();
							for(int i = 0; i < flatStdMat.length; i++) {
								if(flatStdMat[i] == 0) {
									badBehav.set(i, 0, badBehav.get(i, 0) + 1);
								}
							}
							
							badBeh.add(badBehav);
							
							if(countNewTotal > numBootstraps) {
								// Please check behavior data
								throw new Exception();
							}
							
							int[][] reorderP = null;
							if(imagingType == 2)
								reorderP = new RRIBootstrapOrder(st.numSubjectList, st.num_cond_lst[0], 
										numBootstraps, true, minSubjectsPerGroup, isBootstrapSamples, 
										bootstrapSamples, newNumBootstraps, 50.0).result;								
							else
								reorderP = new RRIBootstrapOrder(st.numSubjectList, st.numConditions, 
										numBootstraps, true, minSubjectsPerGroup, isBootstrapSamples, 
										bootstrapSamples, newNumBootstraps, 50.0).result;
							MLFuncs.setColumn(bootOrder, p, MLFuncs.getColumn(reorderP, p));
							
							// Recalculate stdMat
							for(int c = 1; c <= k; c++) {
								int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span);
								int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), 
										range2);
								stdMat.setMatrix(c, c, 0, refPls.stackedBehavData.getColumnDimension() - 1, 
										MLFuncs.std(MLFuncs.getRows(refPls.stackedBehavData, range3)));
							}
						}
						break;
					}
				}
				
				// Now, we can use this proper reorder matrix to generate behav_p & data_p, and then to 
				// calculate datamatcoors
				behavP = MLFuncs.getRows(refPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
				dataP = MLFuncs.getRows(st.datamat, MLFuncs.getColumn(bootOrder, p));

				Matrix Bdata = null;
				
				if(numGroups == 1) {
					Bdata = new RRICorrMapsNotAll(behavP, dataP, n, bscan).maps;
				} else {
					Bdata = new RRICorrMapsNotAll(MLFuncs.getRows(behavP, range), MLFuncs.getRows(dataP, 
							range), n, bscan).maps;
				}
				
				// Matrix TBdata = MLFuncs.append(Tdata, Bdata); needed?
				Matrix data = MLFuncs.append(MLFuncs.normalizeColumn(Tdata), MLFuncs.normalizeColumn(Bdata));
				
				stackedData = MLFuncs.append(stackedData, data);
				progress.endTask(); // curr. bootstrap
			}
			
			SingularValueDecomposition USV = new SingularValueDecomposition(stackedData.transpose());
			
			Matrix pBrainLV = USV.getU();
			Matrix sBoot = USV.getS();
			Matrix pBehavLV = USV.getV();
			
			// Rotate pbehavlv to align with the original behavlv
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.V, pBehavLV).rotatedMatrix;
			
			Matrix temp = sBoot.times(rotatedMatrix);
			pBrainLV = pBrainLV.times(temp);
			pBehavLV = pBehavLV.times(temp);
			
			RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(MLFuncs.getRows(dataP, refPls.rowIdx), 
					MLFuncs.getRows(behavP, refPls.rowIdx), MLFuncs.normalizeRow(pBrainLV),
					MLFuncs.normalizeRow(pBehavLV), kk, st.numSubjectList);
			Matrix bCorr = rgb.lvCorrs;
			
			distribution[p + 1] = bCorr;

			salienceSum.plusEquals(pBrainLV);
			salienceSq.plusEquals(MLFuncs.square(pBrainLV));
			designSalienceSum.plusEquals(pBehavLV);
			designSalienceSq.plusEquals(MLFuncs.square(pBehavLV));
		}

		Matrix salienceSum2 = MLFuncs.square(salienceSum).times(1.0 / (numBootstraps + 1));
		Matrix designSalienceSum2 = MLFuncs.square(designSalienceSum).times(1.0 / (numBootstraps + 1));
		
		// Compute standard errors - standard deviation of bootstrap sample since original sample is part 
		// of bootstrap, divide by number of bootstrap iterations rather than number of bootstraps minus 1
		
		// Add ceiling to calculations to prevent the following operations from producing negative/complex 
		// numbers
		Matrix brainStandardErrors = MLFuncs.sqrt(salienceSq.minus(salienceSum2).times(1.0 / numBootstraps));
		Matrix behavStandardErrors = MLFuncs.sqrt(designSalienceSq.minus(designSalienceSum2).
				times(1.0 / numBootstraps));
		
		int[] brainZeros = MLFuncs.findLessThanOrEqualTo(brainStandardErrors, 0);
		brainStandardErrors = MLFuncs.setValues(brainStandardErrors, brainZeros, 1);
		
		int[] behavZeros = MLFuncs.findLessThanOrEqualTo(behavStandardErrors, 0);
		behavStandardErrors = MLFuncs.setValues(behavStandardErrors, behavZeros, 1);
		
		Matrix compare = originalSal.arrayRightDivide(brainStandardErrors);
		Matrix compareBehavLV = originalBehavLV.arrayRightDivide(behavStandardErrors);
		
		compare = MLFuncs.setValues(compare, brainZeros, 0);
		compareBehavLV = MLFuncs.setValues(compareBehavLV, behavZeros, 0);
		
		double ul = confidenceLimit;
		double ll = 100.0 - confidenceLimit;
		
		// e.g. 0.05 >> 0.025 for upper & lower tails, two-tailed
		double cLimNi = 0.5 * (1 - (confidenceLimit * 0.01));

		Matrix ulCorr = new Matrix(r1, c1);
		Matrix llCorr = new Matrix(r1, c1);
		Matrix prop = new Matrix(r1, c1);
		Matrix ulCorrAdj = new Matrix(r1, c1);
		Matrix llCorrAdj = new Matrix(r1, c1);
		// Loop to calculate upper and lower CI limits
		for(int r = 0; r < r1; r++) {
			for(int c = 0; c < c1; c++) {
				double distrib[] = new double[numBootstraps];
				for(int i = 1; i < numBootstraps + 1; i++) {
					distrib[i - 1] = distribution[i].get(r, c);
				}
				
				Percentile per = new Percentile();
				ulCorr.set(r, c, per.evaluate(distrib, ul));
				llCorr.set(r, c, per.evaluate(distrib, ll));
				prop.set(r, c, ((double)MLFuncs.findLessThanOrEqualTo(distrib, 
						origCorr.get(r, c)).length) / numBootstraps);
				
				if(prop.get(r, c) == 1 || prop.get(r, c) == 0) {
					// Can't calculate the cumulative_gaussian_inv
					llCorrAdj.set(r, c, Double.NaN);
					ulCorrAdj.set(r, c, Double.NaN);
				} else {
					// Adjusted confidence intervals - in case the bootstrap samples are extremely skewed
					
					// Norm inverse to start to adjust conf int
					NormalDistributionImpl normalDistribution = new NormalDistributionImpl();
					double ni = normalDistribution.inverseCumulativeProbability(prop.get(r, c));
					
					// 1st part of recalc the lower conf interval, this evaluates to +1.96 for 95%CI
					double uli = 2 * ni + normalDistribution.inverseCumulativeProbability(1 - cLimNi);
					
					// 1st part of recalc the upper conf interval e.g -1.96 for 95%CI
					double lli = 2 * ni + normalDistribution.inverseCumulativeProbability(cLimNi);
					
					// percentile for lower bounds
					double nCDF_lli =  normalDistribution.cumulativeProbability(lli) * 100;
					
					// percentile for upper bounds
					double nCDF_uli =  normalDistribution.cumulativeProbability(uli) * 100;
					
					// new percentile
					llCorrAdj.set(r, c, per.evaluate(distrib, nCDF_lli));
					ulCorrAdj.set(r, c, per.evaluate(distrib, nCDF_uli));
				}
			}
		}
		
		bootResult = new BootstrapResult();
		bootResult.numBootstraps = numBootstraps;
		bootResult.origCorr = origCorr;
		bootResult.ulCorr = ulCorr;
		bootResult.llCorr = llCorr;
		bootResult.ulCorrAdj = ulCorrAdj;
		bootResult.llCorrAdj = llCorrAdj;
		bootResult.prop = prop;
		bootResult.distrib = distribution;
		bootResult.origBrainLV = originalSal;
		bootResult.brainStandardErrors = brainStandardErrors;
		bootResult.compare = compare;
		bootResult.compareBehavLV = compareBehavLV;
		bootResult.bootSample = bootOrder;
		bootResult.badBeh = badBeh;
		bootResult.countNewTotal = countNewTotal;
		
		/* 
		if(imagingType == 2){
			bootResult.zeroBrainStandarErrors = brainZeros;
			bootResult.numLowVariabilityBehavBoots = numLowVariabilityBehavBoots;
		}
		*/
		
		}
	}
}
