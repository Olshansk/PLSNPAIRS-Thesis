package pls.analysis;

import java.util.Vector;

import pls.shared.MLFuncs;



/**
 * Check to see if the given number of bootstraps is possible.
 * TODO: Pop up gui if new number is needed (Requires a gui).
 * @author imran
 *
 */
public class RRIBootstrapCheck {
	
	protected int minSubjectsPerGroup = 0;
	
	protected final int maxSubjectsPerGroup = 8;
	
	protected boolean[] isBootstrapSamples = null;

	Vector<int[][]> bootstrapSamples = new Vector<int[][]>();
	
	protected int newNumBootstraps = 0;
	
	//	 TODO: This should prompt for new number of bootstraps if necessary when gui is implemented
	public RRIBootstrapCheck(int[] numSubjectList, int numConditions, int numBootstraps, boolean includeSequential, double percentage) throws Exception {
		
		if(numBootstraps == 0) {
			return;
		}
		
		int numGroups = numSubjectList.length;
		
		this.newNumBootstraps = numBootstraps;
		
		System.out.println("RRI bootstrap icinde "+MLFuncs.min(numSubjectList));
		System.out.println("RRI bootstrap icinde "+numSubjectList[0]);
		
		
		if(MLFuncs.min(numSubjectList) < 3) {
			throw new Exception("Number of subjects in one of the groups is less than 3");
		}
		
		this.minSubjectsPerGroup = (int)Math.ceil(MLFuncs.min(numSubjectList) * percentage / 100);
		
		this.isBootstrapSamples = new boolean[numGroups];
		
		if(MLFuncs.numberLessThanOrEqualTo(numSubjectList, this.maxSubjectsPerGroup) == numGroups) {
			for(int g = 0; g < numGroups; g++) {
				int numSubjects = numSubjectList[g];
				
				int[][] bootstrapSample2 = null;
				
				int maxDiffSubject;
				// Remove sequential order
				if(includeSequential) {
					maxDiffSubject = numSubjects;
				} else {
					maxDiffSubject = numSubjects - 1;
				}
				
				for(int diffSubjects = this.minSubjectsPerGroup; diffSubjects <= maxDiffSubject; diffSubjects++) {
					int[][] bootstrapSample1 = new RRIBootstrapSamples(numSubjects, diffSubjects).result;
					bootstrapSample2 = MLFuncs.append(bootstrapSample2, bootstrapSample1);
				}
				
				int numBootstrapSamples = bootstrapSample2.length;
				
				if(numBootstrapSamples < newNumBootstraps) {
					throw new Exception(numSubjects + " subjects can only have " + numBootstrapSamples + " different bootstrap samples.\n" +
							"Please reduce the number of bootstraps or choose another percentage number\n" +
							"(between 30 and 70) that will represent a minimum number of different subjects\n" +
							"in bootstrap versus total number of subjects you have.\n" +
							"If you are not sure, please accept the default value which is equal to 50 percent.");
				}
				
				if(numBootstrapSamples > 0) {
					bootstrapSample2 = MLFuncs.getRows(bootstrapSample2, MLFuncs.randomPermutations(numBootstrapSamples));
					this.bootstrapSamples.add(bootstrapSample2);
					this.isBootstrapSamples[g] = true;
				}
			}
		}
	}
}
