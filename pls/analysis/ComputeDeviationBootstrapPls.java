package pls.analysis;

import java.util.Vector;

import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.stat.descriptive.rank.Percentile;

import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputeDeviationBootstrapPls {
	
	protected BootstrapResult bootResult = null;
	
	public ComputeDeviationBootstrapPls(ComputeDeviationPls refPls, Matrix stDatamat, int numConditions, 
			int[] eventList, int numBootstraps, int[] subjectGroup, int minSubjectsPerGroup, 
			boolean[] isBootstrapSamples, Vector<int[][]> bootstrapSamples, int newNumBootstraps, 
			StreamedProgressHelper progress) throws Exception {
		
		if(numBootstraps == 0) {
			return;
		}
		
		double confidenceLimit = 95;
		
		int numGroups = subjectGroup.length;
		
		int[][] bootOrder = (new RRIBootstrapOrder(subjectGroup, numConditions, numBootstraps, true, 
				minSubjectsPerGroup, isBootstrapSamples, bootstrapSamples, newNumBootstraps, 50.0)).result;
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
		if(MLFuncs.sum(MLFuncs.lessThanOrEqualTo(subjectGroup, subjectGroup.length)) == numGroups) {
			isBootstrapSamples = new boolean[]{true};
		} else {
			isBootstrapSamples = new boolean[]{false};
		}
		
		if(bootOrder.length == 0) {
			return;
		}
		
		Matrix diagS = MLFuncs.diag(refPls.S);
		Matrix originalSaliences = refPls.brainLV.times(diagS);
		Matrix originalDesignSaliences = refPls.designLV.times(diagS);
				
		// Perform bootstrap now
		
		Matrix salienceSum = new Matrix(refPls.brainLV.getRowDimension(), refPls.brainLV.
				getColumnDimension());
		Matrix salienceSq = new Matrix(refPls.brainLV.getRowDimension(), refPls.brainLV.
				getColumnDimension());
		Matrix designSalienceSum = new Matrix(refPls.designLV.getRowDimension(), refPls.designLV.
				getColumnDimension());
		Matrix designSalienceSq = new Matrix(refPls.designLV.getRowDimension(), refPls.designLV.
				getColumnDimension());
		
		//Calculate Brain Scores
		Matrix sMeanmat = new GroupDeviationData(stDatamat, numConditions, eventList, subjectGroup).sMeanmat;
		Matrix bScores2 = sMeanmat.times(refPls.brainLV);
		
		Matrix origUsc = null;
		
		int first = 0;
		int last = -1;
		
		for(int i = 0; i < subjectGroup.length; i++) {
			
			//not sure in what situations you will have subjectGroup[i] as a cell array
			last += numConditions*subjectGroup[i];
			Matrix A = bScores2.getMatrix(first, last, 0, bScores2.getColumnDimension()-1);
			Matrix Means = new RRITaskMean(A, subjectGroup[i]).taskMean;
			origUsc = MLFuncs.append(origUsc,Means);
			first = last + 1; 
		}
					
		Matrix[] distribution = new Matrix[numBootstraps + 1];
		
		distribution[0] = origUsc;
		
		for(int k = 0; k < numBootstraps; k++) {
			progress.startTask("Computing bootstrap no. " + (k + 1), "Bootstrap no, " + (k + 1));
			int[] newOrder = MLFuncs.getColumn(bootOrder, k);
			
			Matrix devData = new GroupDeviationData(MLFuncs.getRows(stDatamat, newOrder), numConditions, 
					eventList, subjectGroup).data;
			Matrix boot_sMeanmat = new GroupDeviationData(MLFuncs.getRows(stDatamat, newOrder), numConditions, eventList, subjectGroup).sMeanmat;
			
			SingularValueDecomposition USV = new SingularValueDecomposition(devData.transpose());
			
			Matrix brainLV = USV.getU();
			Matrix S = USV.getS();
			Matrix designLV = USV.getV();
			
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.designLV, designLV).rotatedMatrix;
			
			Matrix sTimesRM = S.times(rotatedMatrix);
			brainLV = brainLV.times(sTimesRM);
			designLV = designLV.times(sTimesRM);
								
			Matrix boot_bScores2 = boot_sMeanmat.times(MLFuncs.normalizeEuc(brainLV, 1));
			Matrix tmp_origUsc = null;
			
			int f = 0;
			int l = -1;
			
			for(int i = 0; i < subjectGroup.length; i++) {
			    //not sure in what situations you will have subjectGroup[i] as a cell array
				l += numConditions*subjectGroup[i];
				Matrix A = boot_bScores2.getMatrix(f, l, 0, boot_bScores2.getColumnDimension()-1);
				Matrix Means = new RRITaskMean(A, subjectGroup[i]).taskMean;
				tmp_origUsc = MLFuncs.append(tmp_origUsc,Means);
				f = l + 1; 
			}
			
			distribution[k + 1] = tmp_origUsc;
			
			salienceSum.plusEquals(brainLV);
			salienceSq.plusEquals(MLFuncs.square(brainLV));

			designSalienceSum.plusEquals(designLV);
			designSalienceSq.plusEquals(MLFuncs.square(designLV));

			progress.endTask(); // curr. bootstrap
		}
		
		Matrix salienceSum2 = MLFuncs.square(salienceSum).times(1.0 / numBootstraps);
		Matrix brainStandardErrors = MLFuncs.sqrt(
				salienceSq.minus(salienceSum2).times(1.0 / (numBootstraps - 1)));
		
		Matrix designSalienceSum2 = MLFuncs.square(designSalienceSum).times(1.0 / numBootstraps);
		Matrix designStandardErrors = MLFuncs.sqrt(
				designSalienceSq.minus(designSalienceSum2).times(1.0 / (numBootstraps - 1)));
		
		// Check for zero standard errors - replace with ones
		int[] brainZeros = MLFuncs.findLessThanOrEqualTo(brainStandardErrors, 0);
		brainStandardErrors = MLFuncs.setValues(brainStandardErrors, brainZeros, 1);

		Matrix compare = originalSaliences.arrayRightDivide(brainStandardErrors);
		compare = MLFuncs.setValues(compare, brainZeros, 0);
		
		Matrix compareDesign = originalDesignSaliences.arrayRightDivide(designStandardErrors);
		
		//Compute Confidence Interval
		int r1 = origUsc.getRowDimension();
		int c1 = origUsc.getColumnDimension();
		
		double ul = confidenceLimit;
		double ll = 100.0 - confidenceLimit;
		
		// e.g. 0.05 >> 0.025 for upper & lower tails, two-tailed
		double cLimNi = 0.5 * (1 - (confidenceLimit * 0.01));

		Matrix ulUsc = new Matrix(r1, c1);
		Matrix llUsc = new Matrix(r1, c1);
		Matrix prop = new Matrix(r1, c1);
		Matrix ulUscAdj = new Matrix(r1, c1);
		Matrix llUscAdj = new Matrix(r1, c1);
					
		// Loop to calculate upper and lower CI limits
		for(int r = 0; r < r1; r++) {
			for(int c = 0; c < c1; c++) {
				
				double distrib[] = new double[numBootstraps];
				
				for(int i = 1; i < numBootstraps + 1; i++) {
					distrib[i - 1] = distribution[i].get(r, c);
				}
				
				Percentile per = new Percentile();
				ulUsc.set(r, c, per.evaluate(distrib, ul));
				llUsc.set(r, c, per.evaluate(distrib, ll));
				prop.set(r, c, ((double)MLFuncs.findLessThanOrEqualTo(distrib, origUsc.get(r, c)).length) / numBootstraps);
				
				if(prop.get(r, c) == 1 || prop.get(r, c) == 0) {
					// Can't calculate the cumulative_gaussian_inv
					llUscAdj.set(r, c, Double.NaN);
					ulUscAdj.set(r, c, Double.NaN);
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
					llUscAdj.set(r, c, per.evaluate(distrib, nCDF_lli));
					ulUscAdj.set(r, c, per.evaluate(distrib, nCDF_uli));
				}
			}
		}
		
		bootResult = new BootstrapResult();
		bootResult.numBootstraps = numBootstraps;
		bootResult.bootSample = bootOrder;
		bootResult.brainStandardErrors = brainStandardErrors;
		bootResult.designStandardErrors = designStandardErrors;
		bootResult.compare = compare;
		bootResult.compareDesign = compareDesign;
		bootResult.bScores2 = bScores2;
		bootResult.origUsc =origUsc;
		bootResult.ulUsc = ulUsc;
		bootResult.llUsc = llUsc;
		bootResult.ulUscAdj = ulUscAdj;
		bootResult.llUscAdj = llUscAdj;
		bootResult.distrib = distribution;
		bootResult.prop = prop;
			
	}
		
}
