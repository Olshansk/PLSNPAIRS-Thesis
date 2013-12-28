package pls.analysis;

import Jama.Matrix;
import Jama.SingularValueDecomposition;;

public class RRIBootstrapProcrustes {
	
	protected Matrix rotatedMatrix = null;
	
	public RRIBootstrapProcrustes(Matrix origLV, Matrix bootLV) {
		// Define coordinate space between original and bootstrap latent variables
		Matrix temp = origLV.transpose().times(bootLV);
		
		SingularValueDecomposition USV = new SingularValueDecomposition(temp);
		
		this.rotatedMatrix = USV.getV().times(USV.getU().transpose());
	}
}
