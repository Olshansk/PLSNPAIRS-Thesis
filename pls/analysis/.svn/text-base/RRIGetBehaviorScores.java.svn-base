package pls.analysis;

import Jama.Matrix;
import pls.shared.MLFuncs;

public class RRIGetBehaviorScores {
	
	protected Matrix scores = null;
	
	protected Matrix fScores = null;
	
	protected Matrix lvCorrs = null;
	
	public RRIGetBehaviorScores(Matrix stackedDatamat, Matrix stackedBehavData, Matrix brainLV, Matrix behavLV, int k, int[] numSubjectList) {
		scores = stackedDatamat.times(brainLV);
		
		int numGroups = numSubjectList.length;
		
		for(int g = 1; g <= numGroups; g++) {
			int n = numSubjectList[g - 1];
			int t = stackedBehavData.getColumnDimension();
			Matrix temp = null;
			
			int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g - 2))) * k;
			
			for(int i = 1; i <= k; i++) {
				int[] range1 = MLFuncs.range(n * (i - 1) + span, (n * i) + span - 1);
				int[] range2 = MLFuncs.range(t * (i - 1) + (g - 1) * t * k, (t * i + (g - 1) * t * k) - 1);
				Matrix tempK = MLFuncs.getRows(stackedBehavData, range1).times(MLFuncs.getRows(behavLV, range2));
				temp = MLFuncs.append(temp, tempK);
			}
			
			fScores = MLFuncs.append(fScores, temp);

			int[] range = MLFuncs.range(span, n * k + span - 1);
			temp = new RRICorrMaps(MLFuncs.getRows(stackedBehavData, range), MLFuncs.getRows(scores, range), n, k).maps;
			
			lvCorrs = MLFuncs.append(lvCorrs, temp);
		}
	}
}
