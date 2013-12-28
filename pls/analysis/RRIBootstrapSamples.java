package pls.analysis;

import pls.shared.MLFuncs;

public class RRIBootstrapSamples {
	
	protected int[][] result = null;
	
	protected int numBootSamples = 0;

	public RRIBootstrapSamples(int n, int diffSubjects) {
		int[] B = null;
		
		int bootstrapNum = 0;
		int k = n - 1;
		
		int[] A = MLFuncs.ones(n);
		
		while(A[0] < n) {
			int[] array = new int[n];
			
			for(int i = 0; i < n; i++) {
				array[A[i] - 1] = 1;
			}
			
			int sum = 0;
			
			for(int i = 0; i < n; i++) {
				sum += array[i];
			}
			
			if(sum == diffSubjects) {
				for(int i = 0; i < n; i++) {
					B = MLFuncs.append(B, A[i] - 1);
				}
				
				bootstrapNum++;
			}
			
			if(A[k] == n) {
				int i = k;
				
				while(A[i] == n) {
					i--;
				}
				
				A[i]++;
				
				for(int j = i + 1; j < n; j++) {
					A[j] = A[i];
				}
			} else {
				A[k]++;
			}
		}
		
		if(diffSubjects == 1) {
			for(int i = 0; i < n; i++) {
				B = MLFuncs.append(B, n);
			}
			bootstrapNum++;
		}
		
		this.result = MLFuncs.transpose(MLFuncs.reshape(new int[][]{B}, n, B.length/n));
	}

	/** Uril's method
	public RRIBootstrapSamples(int n, int diffSubjects) {
		
		int bootstrapNum = 0; int k = n;
		
		int[] array = new int[n + 1];
		int[] A = new int[n + 1];
		int[][] Permut = new int[2000][n + 1];
		
		for(int i = 1; i <= n; i++) {
			A[i] = 1;
		}
		
		while(A[1] < n) {
			for(int i = 1; i <= n; i++) {
				array[i] = 0;
			}
			for(int i = 1; i <= n; i++) {
				array[A[i]] = 1;
			}
			
			int sum = 0;
			
			for(int i = 1; i <= n; i++) {
				sum += array[i];
			}
			
			if(sum == diffSubjects) {
				for(int i = 1; i <= n; i++) {
					Permut[bootstrapNum][i] = A[i];
				}
				bootstrapNum++;
			}
			
			if(A[k] == n) {
				int i = k;
				
				while(A[i] == n) {
					i--;
				}
				
				A[i]++;
				
				for(int j = i + 1; j <= n; j++) {
					A[j] = A[i];
				}
			} else {
				A[k]++;
			}
		}
		
		if(diffSubjects == 1) {
			bootstrapNum++;
			for(int i = 1; i <= n; i++) {
				Permut[i] = MLFuncs.fillArray(n, n);
			}
			bootstrapNum++;
		}
		
		int[][] Permut_data = new int[bootstrapNum][n];
		
		for(int i = 0; i < bootstrapNum; i++) {
			for(int j = 1; j <= n; j++) {
				Permut_data[i][j - 1] = Permut[i][j];
			}
		}
		this.numBootSamples = bootstrapNum;
		
		// Since indeces start at 0 in Java
		this.result = MLFuncs.minus(Permut_data, 1);
	}
	**/
	
	public static void main(String[] argv) {
		int n = Integer.parseInt(argv[0]);
		int diffSubjects = Integer.parseInt(argv[1]);
		
		int[][] result2 = new RRIBootstrapSamples(n, diffSubjects).result;
		for(int i = 0; i < result2.length; i++) {
			for(int j = 0; j < result2[0].length; j++) {
				System.out.print(result2[i][j] + " ");
			}
			System.out.print("\n");
		}
	}
}
