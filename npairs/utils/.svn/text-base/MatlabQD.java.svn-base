package npairs.utils;

import java.util.Enumeration;
import java.util.Hashtable;

import jmatlink.JMatLink;
import npairs.NpairsjException;
import npairs.shared.matlib.*;
import pls.shared.MLFuncs;

public class MatlabQD extends CVA{

	protected double rmsError = 0;
	//protected Matrix cvaEvalMat;

	protected double[] cvaEvals;
	protected Matrix cvaEvects;	// cva eigenimages before projection into any other basis
	protected Matrix cvaScores;   // contains projection of input data onto cv
								// eigenimages,
								// i.e., representation of input data in cv space;
								// num rows = num rows in input data;
								// num cols = num CV dims
	protected Matrix cvaEigims = null;
//	protected Matrix chiSqrInfo;
//	private double[][] confidReg;

	protected int nSigDim;
	protected Matrix r2; // contains r^2 values calculated for each input data dim and CV dim
	                   // size(r2) = no. input data dims (e.g. no. PC dims) rows X 
	                   // no. CV dims cols

	protected int[] classLabels;
	protected int nCVDim;
	protected int nClasses;
	protected int[] clsSz; // contains class sizes for each class label (class labels
						 // in ascending order)
	protected int nVol;
	protected boolean computeR2 = false; // Recommended: set to false unless you want 
	                                   // to examine R2 output (since it's a non-trivial
	                                   // exercise to compute R2 values).
	protected boolean computeChiSqr = false; // recommended: set to false unless
	                                        // you want to examine chi squared output 
	                                        // (since it's non-trivial to calculate)
	protected boolean cvaInOrigSpace; // true if cva eigims live in original (image) space
	
	// Grigori: new variables
	protected Matrix mean0;  // in "small" PC space
	protected Matrix mean1; 
	protected Matrix inv_cov0;
	protected Matrix inv_cov1;
	protected double logdet0;
	protected double logdet1;
	
	protected Matrix training_data; // in small PC space
	protected Matrix basis; // PC basis (#features x #PCs)
	protected Matrix train_mean0; // in feature space! 
	protected Matrix train_mean1;
	
	
	// Grigori: Matlab engine
	private static JMatLink engine = null;
	private static int matlabMatCount = 0; 	  // counts each instance of MatlabMatrix.
	

	public MatlabQD(Matrix data, int[] labels, Matrix basisVects) throws NpairsjException {
		classLabels = labels;
		training_data = data;
		this.basis = basisVects;
		nVol = classLabels.length;
		nCVDim = 1;
		matlabMatCount += 1;
		if (engine == null) {
			System.out.println("Opening new matlab engine...");
			engine = new JMatLink();
			engine.engOpen();
		}

		engine.engEvalString("cd H:/code/matlab_npairs/QD_for_Java;");
		engine.engEvalString("clear all");
		engine.engPutArray ("labels", MLFuncs.toDoubleArray(labels));
		engine.engPutArray ("data", data.toArray());
		engine.engEvalString("data = data';");
		// create QD
		engine.engEvalString("[mean0 mean1 inv_cov0 inv_cov1 logdet0 logdet1 evalue] = QD (data, labels);");			
		mean0 = new MatrixImpl(engine.engGetArray("mean0")).getMatrix();
		mean1 = new MatrixImpl(engine.engGetArray("mean1")).getMatrix();
		inv_cov0 = new MatrixImpl(engine.engGetArray("inv_cov0")).getMatrix();
		inv_cov1 = new MatrixImpl(engine.engGetArray("inv_cov1")).getMatrix();		
		logdet0 = engine.engGetScalar("logdet0");
		logdet1 = engine.engGetScalar("logdet1");

		// DEBUGGING
		String cmd = "save "+"features"+matlabMatCount+" data;";
		engine.engEvalString(cmd);
		cmd = "save "+"labels"+matlabMatCount+" labels;";
		engine.engEvalString(cmd);
		
		
		
		cvaEvals = new double [1];
		cvaEvals[0] = engine.engGetScalar("evalue");
		
		engine.engEvalString("unique_labels = unique (labels);");
		engine.engEvalString("idx1 = find (labels == unique_labels (1));");
		engine.engEvalString("coord1 = data (:, idx1);");
		engine.engEvalString("mean1 = mean (coord1, 2);");
		
//		engine.engEvalString("sz = size (data);");
//		double[][] sz = engine.engGetArray("sz");
//		System.out.println ("size of mean: " + mean0.numRows()+" x "+mean0.numCols());
//		System.out.println ("size of inv_cov: " + inv_cov0.numRows()+" x "+inv_cov0.numCols());
//		System.out.println ("size of data: " + data.numRows()+" x "+data.numCols());
//		System.out.println ("size of basis: " + basisVects.numRows()+" x "+basisVects.numCols());		
//		System.out.println ("size of data in Matlab:" + sz[0][0]+ "x" + sz[0][1]);
//		engine.engEvalString("sz = size (mean1);");
//		sz = engine.engGetArray("sz");
//		System.out.println ("size of mean in Matlab:" + sz[0][0]+ "x" + sz[0][1]);		
		
		// training set means (in feature space)
		train_mean0 = basis.mult(mean0);  
		train_mean1 = basis.mult(mean1); 
		// training data (in feature space)
		Matrix data_in_feat_space = basis.mult(training_data.transpose());
		// compute training scores
		engine.engPutArray ("train_mean0", train_mean0.toArray());
		engine.engPutArray ("train_mean1", train_mean1.toArray());
		engine.engPutArray ("basis", basis.toArray());		
		engine.engPutArray ("data", data_in_feat_space.toArray());	
		engine.engEvalString("scores = QD_scores (inv_cov0, inv_cov1, logdet0, logdet1, basis, train_mean0, train_mean1, data);");
		cvaScores = new MatrixImpl(engine.engGetArray("scores")).getMatrix().transpose();
//		System.out.println ("size of cvaScores: " + cvaScores.numRows()+" x "+cvaScores.numCols());		
	}

	
	public void createEigenimages(Matrix basisVects, boolean basisVectsInOrigSpace) {
		engine.engEvalString("clear all");
		engine.engPutArray ("data", training_data.toArray());
		engine.engPutArray ("inv_cov0", inv_cov0.toArray());
		engine.engPutArray ("inv_cov1", inv_cov1.toArray());		
		engine.engPutArray ("mean0", mean0.toArray());
		engine.engPutArray ("mean1", mean1.toArray());		
		
		if (basisVects != null) {
			basis = basisVects;
			engine.engPutArray ("basis", basisVects.toArray());					
		} else {
			// basis set is an identity matrix
			engine.engEvalString("basis = eye (length (mean1));");
		}
		engine.engEvalString("data = data';");
		engine.engEvalString("A = inv_cov1 - inv_cov0;");
		engine.engEvalString("b = inv_cov1 * mean1 - inv_cov0 * mean0;");
//		engine.engEvalString("sz = size (data);");
//		double[][]sz = engine.engGetArray("sz");
//		System.out.println ("size of data in Matlab:" + sz[0][0]+ "x" + sz[0][1]);			
		
		engine.engEvalString("map = QD_map (inv_cov0, inv_cov1, mean0, mean1, basis, data);");
		cvaEigims = new MatrixImpl(engine.engGetArray("map")).getMatrix();	
//		System.out.println ("size of cvaEigims: " + cvaEigims.numRows()+" x "+cvaEigims.numCols());
		cvaInOrigSpace = basisVectsInOrigSpace;		
	}
	
	public void rotateEigimsToOrigSpace(Matrix invProjectedData, Matrix origData) {
		if (cvaInOrigSpace) {
			return;
		}
		Matrix P1invP = cvaEigims.transpose().mult(invProjectedData);
		Matrix voxSpaceCVAEigims = P1invP.mult(origData);
		cvaEigims = voxSpaceCVAEigims.transpose();
		cvaInOrigSpace = true;
	}
	
	public double[] getEvals() {
		return cvaEvals;
	}
	
	public Matrix getEigims() {
		return cvaEigims;
	}
	
	public Matrix getCVScores() {
		return cvaScores;
	}
	
	public Matrix calcTestCVScores(Matrix testData) {
		engine.engEvalString("clear all");
		engine.engPutArray ("test_data", testData.toArray());
		engine.engEvalString("test_data = test_data';");
		engine.engPutArray ("inv_cov0", inv_cov0.toArray());
		engine.engPutArray ("inv_cov1", inv_cov1.toArray());		
		engine.engPutArray ("train_mean0", train_mean0.toArray());
		engine.engPutArray ("train_mean1", train_mean1.toArray());
		engine.engPutArray ("logdet0", logdet0);
		engine.engPutArray ("logdet1", logdet1);
		engine.engPutArray ("basis", basis.toArray());
		engine.engEvalString("scores = QD_scores (inv_cov0, inv_cov1, logdet0, logdet1, basis, train_mean0, train_mean1, test_data);");
		Matrix testScores = new MatrixImpl(engine.engGetArray("scores")).getMatrix().transpose();
//		System.out.println ("size of testScores: " + testScores.numRows()+" x "+testScores.numCols());		
		return testScores;
	}
	
	public int getNumCVDims() {
		return nCVDim;
	}
	
	public Matrix avgCVScores() {		
		Hashtable<Integer, int[]> condIndices = CVA.getLabelIndices(classLabels);
		int numCond = condIndices.size();
		int numCVDims = cvaScores.numCols();
		int[] sortedUniqCondLabels = new int[numCond];
		int i = 0;
		for (Enumeration<Integer> uniqCondLabels = condIndices.keys(); uniqCondLabels.hasMoreElements(); ) {
			sortedUniqCondLabels[i] = (Integer)uniqCondLabels.nextElement();
			++i;
		}
		sortedUniqCondLabels = MLFuncs.sortAscending(sortedUniqCondLabels);
		
		MatrixImpl avgScoresImpl = new MatrixImpl(numCond, numCVDims);
		Matrix avgScores = avgScoresImpl.getMatrix();
		for (int cond = 0; cond < numCond; ++cond) {
			int[] currCondIndices = condIndices.get(sortedUniqCondLabels[cond]);
			for (int dim = 0; dim < numCVDims; ++dim) {
				double[] currScores = MLFuncs.getItemsAtIndices(cvaScores.getColumn(dim),
						currCondIndices);
				avgScores.set(cond, dim, MLFuncs.avg(currScores));
			}
		}		
		return avgScores;
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		double[] x1 = { 191, 185, 200, 173, 171, 160, 188, 186, 174, 163, 190,
		174, 201, 190, 182, 184, 177, 178, 186, 211, 201, 242, 184,
		211, 217, 223, 208, 199, 211, 218, 203, 192, 195, 211, 187, 192 };
		double[] x2 = { 131, 134, 137, 127, 118, 118, 134, 139, 131, 115, 143,
				131, 130, 133, 130, 131, 127, 126, 107, 122, 114, 131, 108,
				118, 122, 127, 125, 124, 129, 126, 122, 116, 123, 122, 123, 109 };
		double[][] data = new double[2][36];
		data[0] = x1;
		data[1] = x2;
		
		double [] labels = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };			
		
		try
		{
		Matrix M = new MatrixImpl(data, "COLT").getMatrix();
		//M = M.transpose();
		engine = new JMatLink();
		engine.engOpen();		
		engine.engPutArray ("labels", labels);
		engine.engPutArray ("data", M.toArray());
		engine.engEvalString ("idx1 = find (labels == 1);");
		engine.engEvalString ("idx2 = find (labels == 2);");
		engine.engEvalString("cd H:/code/matlab_npairs/QD_for_Java;");
		engine.engEvalString("[mean0 mean1 inv_cov0 inv_cov1 logdet0 logdet1 evalue] = QD (data, labels);");
		//engine.engEvalString("evalue = QD_eigvalue (inv_cov0, inv_cov1, mean0, mean1, logdet0, logdet1);");
		
		//engine.engEvalString("QD_script;");
		double N = engine.engGetScalar("evalue");
		System.out.println (N);
		//engine.engEvalString("temp = S0_cov (1, :);");
		
		engine.engEvalString("basis = eye (length (mean1));");
		engine.engEvalString("[pp1 pp2] = QD_postprob (inv_cov0, inv_cov1, logdet0, logdet1, basis, mean0, mean1, data);");
		Matrix temp2 = new MatrixImpl (engine.engGetArray("inv_cov0")).getMatrix();
		temp2.print();
//		for (int i = 0; i < temp.length; i ++) {
//			for (int j = 0; j < temp[i].length; j ++) {
//				System.out.print (temp[i][j]);
//				System.out.print (" ");
//			}
//			System.out.println ("");
//		}
		
		engine.engClose();		
		
		}
		catch (MatrixException e) {
			System.out.println ("Wrong Matrix Type");
		}
		
		
	


		System.out.println ("That's it folks");
	}

}
