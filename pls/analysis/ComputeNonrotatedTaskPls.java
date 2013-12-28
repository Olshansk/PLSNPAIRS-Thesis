package pls.analysis;

import pls.shared.MLFuncs;
import Jama.Matrix;

public class ComputeNonrotatedTaskPls {
	
	protected Matrix brainLV = null;
	
	protected Matrix S = null;
	
	protected Matrix designLV = null;
	
	protected Matrix brainScores = null;
	
	protected Matrix designScores = null;
	
	protected Matrix LVInterCorrs = null;
	
	protected Matrix crossBlock = null;
	
	protected Matrix sMeanmat = null;
	
	protected Matrix bScores2 = null;
		
	//constructor for fMRI
	public ComputeNonrotatedTaskPls(Matrix stDatamat, int numConditions, int[] eventList, int[] subjectGroup, Matrix design) {
		GroupDeviationData dev = new GroupDeviationData(stDatamat, numConditions, eventList, subjectGroup, design);
		
		sMeanmat = new GroupDeviationData(stDatamat, numConditions, eventList, subjectGroup).sMeanmat;
		
		crossBlock = MLFuncs.normalizeEuc(dev.design, 1).transpose().times(dev.data);
		
		brainLV = crossBlock.transpose();
		
		S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
		
		designLV = dev.design;
		
		Matrix normalizedBrainLV = MLFuncs.normalizeRow(brainLV);
		
		LVInterCorrs = normalizedBrainLV.transpose().times(normalizedBrainLV);
		
		designScores = new ExpandDesignScores(designLV, numConditions, eventList, subjectGroup).newDesignScores;
		
 		brainScores = stDatamat.times(normalizedBrainLV);
		
		bScores2 = sMeanmat.times(normalizedBrainLV);
			
	}
	//constructor for only PET
	public ComputeNonrotatedTaskPls(ConcatenateDatamat st, Matrix design) {
		
		GroupDeviationData dev = new GroupDeviationData(st, design);
		// dev contains data(stacked_data) and design (stacked_designdata
		crossBlock = MLFuncs.normalizeEuc(dev.design, 1).transpose().times(dev.data);
		
		brainLV = crossBlock.transpose();
		
		S = MLFuncs.sqrt(MLFuncs.rowSum(MLFuncs.square(crossBlock)));
		
		designLV = dev.design;
		
		Matrix normalizedBrainLV = MLFuncs.normalizeRow(brainLV);
		
		LVInterCorrs = normalizedBrainLV.transpose().times(normalizedBrainLV);
		
		designScores = new ExpandDesignScores(designLV,  st.numSubjectList, st.num_cond_lst, st.numGroups).newDesignScores;
		
		brainScores = st.datamat.times(brainLV);//st.datamat contains stacked_datamat  
	}
}
