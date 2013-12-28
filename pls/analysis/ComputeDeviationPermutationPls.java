package pls.analysis;

//import java.io.IOException;
//
//import npairs.io.NpairsjIO;
//
//import com.jmatio.types.MLDouble;
//import com.jmatio.types.MLStructure;
//
//import extern.NewMatFileReader;

import npairs.io.NpairsjIO;
import pls.shared.MLFuncs;
import pls.shared.StreamedProgressHelper;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputeDeviationPermutationPls {
	
	protected PermutationResult permResult = null;
	
	public ComputeDeviationPermutationPls(ComputeDeviationPls refPls, Matrix stDatamat, 
			int numConditions, int[] eventList, int numPermutations, int[] subjectGroup, 
			StreamedProgressHelper progress) {
		
		if(numPermutations == 0) {
			return;
		}
		
		// Generate the permutation orders
		int[][] permOrder = new RRIPermutationOrder(subjectGroup, numConditions, numPermutations).result;
//		try {
//			NpairsjIO.printToIDLFile(permOrder, "./permOrderJavaOrigRRIPermOrder.idl");
//		} catch (Exception e) {e.printStackTrace();}
//		MLDouble permOrderStruct = null;
//		try {
//			permOrderStruct = (MLDouble)new NewMatFileReader("/haier/anita/plsnpairs_testing/perm_order.mat").
//				getContent().get("perm_order");
//		} catch (Exception e) {e.printStackTrace(); System.exit(1);}
//		int[][] permOrder = permOrderStruct.getIntArray();
//		permOrder = MLFuncs.minus(permOrder, 1);
		
		// Perform permutation test now
		Matrix sp = new Matrix(1, refPls.S.getColumnDimension());
		Matrix dp = new Matrix(refPls.designLV.getRowDimension(), refPls.designLV.getColumnDimension());
		
		for(int k = 0; k < numPermutations; k++) {
			progress.startTask("Computing permutation no. " + (k + 1), "Perm. no. " + (k + 1));
			int[] newOrder = MLFuncs.getColumn(permOrder, k);
			
			Matrix devData = new GroupDeviationData(MLFuncs.getRows(stDatamat, newOrder), numConditions,  
					eventList, subjectGroup).data;
			
			SingularValueDecomposition USV = new SingularValueDecomposition(devData.transpose());

			Matrix S = MLFuncs.diag(USV.getS());
			Matrix designLV = USV.getV();
			
			Matrix rotatedMatrix = new RRIBootstrapProcrustes(refPls.designLV, designLV).rotatedMatrix;
			
			designLV = designLV.times(MLFuncs.diag(S).times(rotatedMatrix));
			S = MLFuncs.sqrt(MLFuncs.columnSum(MLFuncs.square(designLV)));
			
//			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(S.transpose(), MLFuncs.diag(refPls.S)));
			sp.plusEquals(MLFuncs.greaterThanOrEqualTo(S, refPls.S));
			dp.plusEquals(MLFuncs.greaterThanOrEqualTo(MLFuncs.abs(designLV), MLFuncs.abs(refPls.
					designLV.times(MLFuncs.diag(refPls.S)))));
			
			progress.endTask(); // curr. permutation
		}
		
		sp = sp.transpose(); 
		Matrix sProb = sp.times(1.0 / numPermutations);
		Matrix designLVprob = dp.times(1.0 / numPermutations);

		permResult = new PermutationResult();
		permResult.numPermutations = numPermutations;
		permResult.permSample = permOrder;
		permResult.sp = sp;
		permResult.sProb = sProb;
		permResult.dp = dp;
		permResult.designLVprob = designLVprob;
	}
}
