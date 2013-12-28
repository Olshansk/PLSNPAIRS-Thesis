package pls.analysis;

import java.util.Vector;

import pls.shared.MLFuncs;
import Jama.Matrix;

public class ComputeNonRotatedBehavPls {
	
	protected Vector<Matrix> behavDataList = new Vector<Matrix>();
	
	protected Matrix brainLV = null;
	
	protected Matrix S = null;
	
	protected Matrix behavLV = null;
	
	protected Matrix brainScores = null;
	
	protected Matrix behavScores = null;
	
	protected Matrix LVInterCorrs = null;
	
	protected Matrix lvCorrs = null;
	
	protected Vector<Matrix> datamatCorrsList = new Vector<Matrix>();
	
	protected Matrix crossBlock = null;
	
	protected Matrix stackedBehavData = null;
	
	protected Matrix stackedDatamatCorrs = null;
	
	//constructor for fMRI
	public ComputeNonRotatedBehavPls(Matrix stDatamat, Matrix behavData, Vector<Matrix> behavDataList, Vector<Matrix> newDataList, int[] numSubjectList, int numConditions, int[] eventList, int[] subjectGroup, Matrix design) {
		
		Matrix stackedDatamatCorrs = null;
		stackedBehavData = behavData.copy();
		this.behavDataList = behavDataList;
		
		int numGroups = newDataList.size();
		
		int k = numConditions;
		
		// loop across the groups, and calculate datamatcorrs for each group
		for(int i = 0; i < numGroups; i++) {
			int n = numSubjectList[i];
			Matrix datamat = newDataList.get(i);
			// Compute correlation
			Matrix datamatCorrs = new RRICorrMaps(behavDataList.get(i), datamat, n, k).maps;
			stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
			datamatCorrsList.add(datamatCorrs);
		}
		
		crossBlock = MLFuncs.normalizeEuc(design, 1).transpose().times(stackedDatamatCorrs);
		
		brainLV = crossBlock.transpose();
		
		S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
		
		behavLV = design;
		
		Matrix normalizedBrainLV = MLFuncs.normalizeEuc(brainLV, 1);
		
		LVInterCorrs = normalizedBrainLV.transpose().times(normalizedBrainLV);
		
		
 		
 	// Calculate behav scores
		RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(stDatamat, stackedBehavData, normalizedBrainLV, behavLV, numConditions, numSubjectList);
		brainScores = rgb.scores;
		behavScores = rgb.fScores;
		lvCorrs = rgb.lvCorrs;
					
	}

}
