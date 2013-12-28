package pls.analysis;

import java.util.Vector;

import pls.shared.MLFuncs;

public class RRIBootstrapOrder {
	
	protected int[][] result = null;
	
	protected int newNumBootstraps = 0;
	
	private final int maxSubjectsPerGroup = 8;
	
	public RRIBootstrapOrder(int[] numSubjectList, int numConditions, int numBootstraps, boolean includeSequential, int minSubjectsPerGroup, boolean[] isBootstrapSamples, Vector<int[][]> bootstrapSamples, int newNumBootstraps, double percentage) throws Exception {
		int totalSubjects = MLFuncs.sum(numSubjectList);
		int numGroups = numSubjectList.length;
		int totalRows = numConditions * totalSubjects;
		
		if(minSubjectsPerGroup == 0) { // Can this ever happen?  Do some testing here, I don't think it makes sense
			
			if(MLFuncs.min(numSubjectList) < 3) {
				throw new Exception("Number of subjects in at least one group is less than 3");
			}
			
			if(MLFuncs.sum(MLFuncs.lessThanOrEqualTo(numSubjectList, maxSubjectsPerGroup)) == numGroups) {
				for(int g = 0; g < numGroups; g++) {
					int numSubject = numSubjectList[g];
					
					int[][] bootSample2 = null;
					
					// Remove sequential order
					int maxDiffSubjects = 0;
					if(includeSequential) {
						maxDiffSubjects = numSubject;
					} else {
						maxDiffSubjects = numSubject - 1;
					}
					
					for(int diffSubjects = minSubjectsPerGroup; diffSubjects < maxDiffSubjects; diffSubjects++) {
						int[][] bootSample1 = new RRIBootstrapSamples(numSubject, diffSubjects).result;
						bootSample2 = MLFuncs.append(bootSample2, bootSample1);
					}
					
					int numBootSamples = bootSample2.length;
					
					if(numBootSamples < newNumBootstraps) {
						// Too many bootstraps, user should select less.
						// TODO: This should prompt for different number and redo this procedure
						throw new Exception(numSubject + " subjects can only have " + numBootSamples + " different bootstrap samples.\n" +
								"Please reduce the number of bootstraps or choose another percentage number\n" +
								"(between 30 and 70) that will represent a minimum number of different subjects\n" +
								"in bootstrap versus total number of subjects you have.\n" +
								"If you are not sure, please accept the default value which is equal to 50 percent.");
					}
					
					bootSample2 = MLFuncs.getRows(bootSample2, MLFuncs.randomPermutations(numBootSamples));
					bootstrapSamples.add(bootSample2);
					isBootstrapSamples[g] = true;
				}
			}
		}
		
		// Determine tempBootOrder, which is re-ordered subject index matrix
		int[][] tempBootOrder = new int[totalSubjects][newNumBootstraps];
			
		for(int p = 0; p < newNumBootstraps; p++) {
			Vector<int[]> subjectOrder = new Vector<int[]>(numGroups);
			boolean notDone = true;
			int count = 0;
			
			while(notDone) {
				int startSubject = 1;
				
				for(int g = 0; g < numGroups; g++) {
					int numSubjects = numSubjectList[g];
					
					// Reorder all tasks for the current group
					boolean allSamplesAreSame = true;
					int[] newSubjectOrder = null;
					while(allSamplesAreSame) {
						
						if(isBootstrapSamples[g]) { // Get from boot samples
							newSubjectOrder = bootstrapSamples.get(g)[p];
							allSamplesAreSame = false;
							notDone = false;
						} else {
							notDone = true;
							newSubjectOrder = MLFuncs.randInts(numSubjects, numSubjects);
							int test = MLFuncs.unique(newSubjectOrder).length;
							
							// Check to make sure there are at lease min_subj_per_group people
							if(test >= minSubjectsPerGroup) {
								allSamplesAreSame = false;
							}
						}
					}
					subjectOrder.add(g, MLFuncs.plus(newSubjectOrder, startSubject - 1));
					startSubject += numSubjects;
				}
				
				if(!MLFuncs.all(isBootstrapSamples)) {
					// Make sure the the order is not a repeated one
					notDone = false;
					for(int i = 0; i < p - 1; i++) {
						if(MLFuncs.isEqual(MLFuncs.getColumn(tempBootOrder, i), (subjectOrder.get(0))) && subjectOrder.size() == 1) {
							notDone = true;
						}
					}
					
					// Treat sequential order as duplicated one
					if(!includeSequential && MLFuncs.isEqual(MLFuncs.range(1, totalRows), (MLFuncs.vectorToIntArray(subjectOrder)))) {
						notDone = true;
					}
					
					count++;
					if(count > 500) {
						notDone = false;
						System.out.println("WARNING:  Duplicated bootstrap orders are used!");
					}
				}
				tempBootOrder = MLFuncs.setColumn(tempBootOrder, p, MLFuncs.vectorToIntArray(subjectOrder));
			}
		}
		
		// Construct the resampling order matrix for bootstrap
		
		int[][] rowIdx = null;
		int first = 0, last = -1;
		for(int g = 0; g < numSubjectList.length; g++) {
			last += numConditions * numSubjectList[g];
			int[][] tmp = MLFuncs.reshape(new int[][]{MLFuncs.range(first, last)}, numSubjectList[g], numConditions);
			rowIdx = MLFuncs.appendColumn(rowIdx, MLFuncs.transpose(tmp));
			first = last + 1;
		}
		
		int[][][] bootOrder = new int[newNumBootstraps][numConditions][totalSubjects];
		for(int p = 0; p < newNumBootstraps; p++) {
			for(int j = 0; j < totalSubjects; j++) {
				for(int i = 0; i < numConditions; i++) {
					bootOrder[p][i][j] = rowIdx[i][tempBootOrder[j][p]];
				}
			}
		}
		int[][] bOrder = null;
		for(int g = 0; g < numSubjectList.length; g++) {
			int[][] oneGroup = null;
			for(int p = 0; p < newNumBootstraps; p++) {
				int start = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g - 1)));
				int end = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g))) - 1;
				int[][] tmp = MLFuncs.getColumns(bootOrder[p], MLFuncs.range(start, end));
				tmp = MLFuncs.reshape(MLFuncs.transpose(tmp), numConditions * numSubjectList[g], 1);
				oneGroup = MLFuncs.appendColumn(oneGroup, tmp);
			}
			bOrder = MLFuncs.append(bOrder, oneGroup);
		}
		this.result = bOrder;
		this.newNumBootstraps = newNumBootstraps;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] argv) throws Exception {
		
		int[][] result = null;
		
		if(true) {
			// matlab: rri_boot_order([10, 10, 10], 7, 100, 1, 5, [0, 0, 0], A, 100, 50)
			result = new RRIBootstrapOrder(new int[]{10, 10, 10}, 7, 100, true, 5, new boolean[]{false, false, false}, new Vector<int[][]>(), 100, 50.0).result;
		} else {
			// matlab: rri_boot_order([20], 3, 100, 1, 5, [0], A, 100, 50)
			result = new RRIBootstrapOrder(new int[]{20}, 3, 100, true, 10, new boolean[]{false}, new Vector<int[][]>(), 100, 50.0).result;
		}
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				System.out.print(result[i][j] + "\t");
			}
			System.out.print("\n");
		}
	}
}