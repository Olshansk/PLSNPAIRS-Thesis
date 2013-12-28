package pls.analysis;

import pls.shared.MLFuncs;


public class RRIPermutationOrder {

	protected int[][] result = null;
	
	public RRIPermutationOrder(int[] numSubjectList, int numConditions, int numPermutations) {
		result = new RRIPermutationOrder(numSubjectList, numConditions, numPermutations, false).result;
	}
	public RRIPermutationOrder(int[] numSubjectList, int numConditions, int numPermutations, boolean notInCond) {
		int numSubjectGroups = MLFuncs.sum(numSubjectList);
		int totalRows = numSubjectGroups * numConditions;

		int[][] originalTaskGroup = null;
		
		int[][] permutationOrder = new int[totalRows][numPermutations];
		
		for(int p = 0; p < numPermutations; p++) {
			
			int count = -1;
			boolean duplicated = true;
			int[] newPermutationOrder = null;
			
			while(duplicated) {
				count++;
				
				int first = 0;
				int last = -1;
				int [][] taskGroup = null;
				
				for(int g = 0; g < numSubjectList.length; g++) {
					last += numConditions * numSubjectList[g];
					int[][] tmp = MLFuncs.reshape(MLFuncs.range(first, last), numSubjectList[g], numConditions);
					taskGroup = MLFuncs.appendColumn(taskGroup, MLFuncs.transpose(tmp));
					first = last + 1;
				}
				originalTaskGroup = MLFuncs.copy(taskGroup);
				
				// Exclude this block (to swap conditions for each subject) for structure Non-Behavior PLS
				if(!notInCond) {
					
					int[][] newTaskGroup = new int [numConditions][numSubjectGroups];
					
					// Permute tasks within each group
					for(int i = 0; i < numSubjectGroups; i++) {
						int[] taskPermutations = MLFuncs.randomPermutations(numConditions);
						for(int j = 0; j < taskGroup.length; j++) {
							newTaskGroup[j][i] = taskGroup[taskPermutations[j]][i];
						}
					}
				taskGroup = newTaskGroup;
				
				}
				
				
				// Permute tasks across groups
				int[] groupPermutations = MLFuncs.randomPermutations(numSubjectGroups);
				taskGroup = MLFuncs.getColumns(taskGroup, groupPermutations);
				
				// Make sure the average is not the same as original for cond and for group.  Will mark it as duplicate for invalid one
				duplicated = false;
				
				for(int c = 0; c < numConditions; c++) {
					int accum = -1;
					
					for(int g = 0; g < numSubjectList.length; g++) {
						int[] a = MLFuncs.sortAscending(MLFuncs.getRow(MLFuncs.getColumns(taskGroup, MLFuncs.range(accum + 1, accum + numSubjectList[g])), c));
						int[] b = MLFuncs.getRow(MLFuncs.getColumns(originalTaskGroup, MLFuncs.range(accum + 1, accum + numSubjectList[g])), c);
						boolean different = false;
						for(int i1 = 0; i1 < a.length; i1++) {
							if(a[i1] != b[i1]) {
								different = true;
								break;
							}
							if(different) break;
						}
						if(!different) { // Condition average is the same as the original
							duplicated = true;
						}
						
						accum += numSubjectList[g];
					}
				}
				
				newPermutationOrder = null;
				for(int g = 0; g < numSubjectList.length; g++) {
					int from = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g - 1)));
					int to = MLFuncs.sum(MLFuncs.getItemsAtIndices(numSubjectList, MLFuncs.range(0, g))) - 1;
					int[][] tmp = MLFuncs.getColumns(taskGroup, MLFuncs.range(from, to));
					tmp = MLFuncs.reshape(MLFuncs.transpose(tmp), numConditions * numSubjectList[g], 1);
					newPermutationOrder = MLFuncs.append(newPermutationOrder, MLFuncs.getColumn(tmp, 0));
				}
				
				// Make sure the permutation order is not a repeated one
				for(int i = 0; i < p - 1; i++) {
					int[] a = MLFuncs.getColumn(permutationOrder, i);
					boolean different = false;
					for(int j = 0; j < newPermutationOrder.length; j++) {
						if(a[j] != newPermutationOrder[j]) {
							different = true;
							break;
						}
					}
					if(!different) {
						duplicated = true;
					}
				}
				
				// Treat sequential order as duplicated one
				boolean sequential = true;
				for(int i = 0; i < newPermutationOrder.length; i++) {
					if(i != newPermutationOrder[i]) {
						sequential = false;
						break;
					}
				}
				if(sequential) duplicated = true;
				
				if(count > 500) {
					duplicated = false;
					System.err.println("Error: Duplicated permutation orders are used");
				}
			}
			
			for(int i = 0; i < newPermutationOrder.length; i++) {
				permutationOrder[i][p] = newPermutationOrder[i];
			}
		}
		result = permutationOrder;
	}
	/**
	 * @param args
	 */
	public static void main(String[] argv) {
		int[][] result = new RRIPermutationOrder(new int[]{12, 12}, 5, 500).result;
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				System.out.print(result[i][j] + "\t");
			}
			System.out.print("\n");
		}
	}

}