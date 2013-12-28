package pls.analysis;

import pls.shared.MLFuncs;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

/**
 * Computes a Partial Least Squares Analysis
 * @author imran
 *
 */
public class ComputeDeviationPls {
	
	protected Matrix brainLV = null;
	
	protected Matrix S = null;
	
	protected Matrix designLV = null;
	
	protected Matrix brainScores = null;
	
	protected Matrix designScores = null;
	
	public ComputeDeviationPls(Matrix stDatamat, int numConditions, int[] eventList, int[] subjectGroup) {
		Matrix devData = new GroupDeviationData(stDatamat, numConditions, eventList, subjectGroup).data;
		
		SingularValueDecomposition USV = new SingularValueDecomposition(devData.transpose());
		
		brainLV = USV.getU();
		S = MLFuncs.diag(USV.getS());
		designLV = USV.getV();
		
		designScores = new ExpandDesignScores(designLV, numConditions, eventList, subjectGroup).newDesignScores;

		brainScores = stDatamat.times(brainLV);
	}
}
