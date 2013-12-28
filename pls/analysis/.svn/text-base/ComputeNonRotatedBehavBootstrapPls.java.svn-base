package pls.analysis;

import java.util.Vector;

import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.stat.descriptive.rank.Percentile;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import pls.analysis.RRIIsLowVariability;

import Jama.Matrix;

public class ComputeNonRotatedBehavBootstrapPls {
protected BootstrapResult bootResult = null;
	
	//constructor for fMRI
	public ComputeNonRotatedBehavBootstrapPls(ComputeNonRotatedBehavPls mainPls, Matrix stDatamat, 
			int numConditions, int[] eventList, int numBootstraps, int[] numSubjectList, int[] subjectGroup, 
			int minSubjectsPerGroup, boolean[] isBootstrapSamples, Vector<int[][]> bootstrapSamples, 
			int newNumBootstraps, Matrix design, StreamedProgressHelper progress) throws Exception {
		
		if(numBootstraps == 0) {
			return;
		}
		
		double confidenceLimit = 95;
				
		int k = numConditions;
		int numGroups = subjectGroup.length;
				
		int[][] bootOrder = (new RRIBootstrapOrder(subjectGroup, numConditions, numBootstraps, false, 
				minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0)).result;
		
		// Keeps track of number of times a new bootstrap had to be generated
		int countNewTotal = 0;
		Vector<Matrix> badBeh = new Vector<Matrix>();
		
		Matrix origCorr = null;
		origCorr = mainPls.lvCorrs;
		
		Matrix[] distribution = new Matrix[numBootstraps + 1];

		distribution[0] = origCorr;
				
		Matrix brainLVsq = MLFuncs.square(mainPls.brainLV);
		Matrix brainLVsum = mainPls.brainLV.copy();
		
		//check min% unique values for all behavior variables
		//
		Matrix numLowVar = new Matrix(1, mainPls.stackedBehavData.getColumnDimension());
			
		for(int bw = 0; bw < mainPls.stackedBehavData.getColumnDimension(); bw ++) {
			for(int p = 0; p < numBootstraps; p++) {
				
				Matrix V = MLFuncs.getRows(mainPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
				double[] v = MLFuncs.getColumn(V, bw);
													
				RRIIsLowVariability b = new RRIIsLowVariability(v, MLFuncs.getColumn(mainPls.stackedBehavData, bw));
				
				if (b.status == 1){
					  numLowVar.set(0, bw, numLowVar.get(0, bw) + 1);
				}
			}
		}	
		
		if(MLFuncs.any(numLowVar) == true){
			System.out.println("Warning: For at least one behavior measure, the minimum unique values of resampled behavior data does not exceed 50% of its total.");
		}
					                
		for(int p = 0; p < numBootstraps; p++) {
			
			progress.startTask("Computing bootstrap no. " + (p + 1), "Bootstrap no. " + (p + 1));
			Matrix dataP = MLFuncs.getRows(stDatamat, MLFuncs.getColumn(bootOrder, p));
			Matrix behavP = MLFuncs.getRows(mainPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
			
			Matrix stackedData = null;
						
			for(int g = 1; g <= numGroups; g++) {
				
				int n = numSubjectList[g - 1];
				int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g - 2))) * k;
				int[] range = MLFuncs.range(span, n * k + span - 1);
				
				for(boolean b : isBootstrapSamples) {
					if(!b) {
						// The code below is mainly trying to find a proper reorder matrix
						
						// Init badbehav cell array to 0 which is used to record the bad behav data 
						// caused by bad re-order. This var. is for disp only.
						Matrix badBehav = new Matrix(numConditions, mainPls.behavDataList.get(g - 1).getColumnDimension());
						
						// Check for upcoming NaN and re-sample if necessary. This only happened on 
						// behavior analysis, because the 'xcor' inside of 'corr_maps' contains a 
						// 'stdev', which is a divident. If it is 0, it will cause divided by 0 problem. 
						// Since this happend very rarely, so the speed will not be affected that much.
						
						// For behavpls_boot, also need to account for multiple scans and behavs
						Matrix stdMat = new Matrix(k, mainPls.stackedBehavData.getColumnDimension());
						for(int c = 1; c <= k; c++) {
							int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span - 1);
							int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), range2);
							stdMat.setMatrix(c - 1, c - 1, 0, mainPls.stackedBehavData.getColumnDimension() - 1, MLFuncs.std(MLFuncs.getRows(mainPls.stackedBehavData, range3)));
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
							throw new Exception("Please check behavior data");
						}
							
						int[][] reorderP = new RRIBootstrapOrder(numSubjectList, numConditions, numBootstraps, true, minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0).result;
						MLFuncs.setColumn(bootOrder, p, MLFuncs.getColumn(reorderP, p));
							
						// Recalculate stdMat
						for(int c = 1; c <= k; c++) {
								int[] range2 = MLFuncs.range(n * (c - 1) + span, n * c + span - 1);
								int[] range3 = MLFuncs.getItemsAtIndices(MLFuncs.getColumn(bootOrder, p), range2);
								stdMat.setMatrix(c - 1, c - 1, 0, mainPls.stackedBehavData.getColumnDimension() - 1, MLFuncs.std(MLFuncs.getRows(mainPls.stackedBehavData, range3)));
						}
					}
					
					break;
					
					}
				}	
			
				
				// Now, we can use this proper reorder matrix to generate behav_p & data_p, and then 
				// to calculate datamatcoors
				behavP = MLFuncs.getRows(mainPls.stackedBehavData, MLFuncs.getColumn(bootOrder, p));
				dataP = MLFuncs.getRows(stDatamat, MLFuncs.getColumn(bootOrder, p));

				Matrix data = null;
				
				if(numGroups == 1) {
					data = new RRICorrMaps(behavP, dataP, n, k).maps;
				} else {
					data = new RRICorrMaps(MLFuncs.getRows(behavP, range), MLFuncs.getRows(dataP, range), n, k).maps;
				}		
				stackedData = MLFuncs.append(stackedData, data);
							
			}
							
			Matrix crossBlock = MLFuncs.normalizeEuc(design, 1).transpose().times(stackedData);
			
			RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(dataP, behavP, MLFuncs.normalizeEuc((crossBlock).transpose(), 1), MLFuncs.normalizeRow(mainPls.behavLV), numConditions, numSubjectList);
			Matrix bCorr = rgb.lvCorrs;
			
			distribution[p + 1] = bCorr;

			brainLVsq.plusEquals(MLFuncs.square(crossBlock).transpose());
			brainLVsum.plusEquals(crossBlock.transpose());
			
			progress.endTask(); // curr. bootstrap
		}
					
		Matrix brainLVsum2 = MLFuncs.square(brainLVsum).times(1.0 / (numBootstraps + 1));
		Matrix brainLVStandardErrors = MLFuncs.sqrt((brainLVsq.minus(brainLVsum2)).
					times(1.0 / numBootstraps));
		
		//check for zero standard errors -replace with ones	
			
		int[] brainZeros = MLFuncs.findLessThanOrEqualTo(brainLVStandardErrors, 0);
		brainLVStandardErrors = MLFuncs.setValues(brainLVStandardErrors, brainZeros, 1);
					
		Matrix compare = mainPls.brainLV.arrayRightDivide(brainLVStandardErrors);
			
		compare = MLFuncs.setValues(compare, brainZeros, 0);
		
			
			//Compute Confidence Interval
			int r1 = origCorr.getRowDimension();
			int c1 = origCorr.getColumnDimension();
			
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
					prop.set(r, c, ((double)MLFuncs.findLessThanOrEqualTo(distrib, origCorr.get(r, c)).length) / numBootstraps);
					
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
			bootResult.bootSample = bootOrder;
			bootResult.brainLVStandardErrors = brainLVStandardErrors;
			bootResult.compare = compare;
			bootResult.origCorr = origCorr;
			bootResult.ulCorr = ulCorr;
			bootResult.llCorr = llCorr;
			bootResult.ulCorrAdj = ulCorrAdj;
			bootResult.llCorrAdj = llCorrAdj;
			bootResult.prop = prop;
			bootResult.distrib = distribution;
			bootResult.badBeh = badBeh;
			bootResult.countNewTotal = countNewTotal;
			bootResult.numLowVarBehavBoots = numLowVar;
	}	
}	
				

