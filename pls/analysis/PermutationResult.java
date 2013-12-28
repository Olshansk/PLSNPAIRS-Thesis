package pls.analysis;

import Jama.Matrix;

public final class PermutationResult {
	protected int numPermutations = 0;
	protected int[][] permSample = null;
	protected Matrix sp = null;
	protected Matrix sProb = null;  // These two are equivalent but for some reason matlab code
	protected Matrix sProb2 = null; // used different variable name (saved to file) for behavPLS
	protected Matrix dp = null;
	protected Matrix designLVprob = null;
	protected Matrix vProb = null;
	protected int[][] TpermSamp = null;
	protected int[][] BPermSamp = null;
	protected Matrix posthocProb = null;
}
