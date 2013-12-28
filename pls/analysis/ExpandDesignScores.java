package pls.analysis;

import pls.shared.MLFuncs;
import Jama.Matrix;
import java.util.Vector;

public class ExpandDesignScores {
	Matrix newDesignScores = null;
	
	public ExpandDesignScores(Matrix designScores, int numConditions, int[] eventList, int[] subjectGroup) {
		
		int[] numInGroup = MLFuncs.prepend(0, MLFuncs.product(subjectGroup, numConditions));
		
		for(int groupIdx = 0; groupIdx < subjectGroup.length; groupIdx++) {
			int first = MLFuncs.sum(MLFuncs.getItemsAtIndices(numInGroup, MLFuncs.range(0, groupIdx)));
			int last = MLFuncs.sum(MLFuncs.getItemsAtIndices(numInGroup, MLFuncs.range(0, groupIdx + 1)));
			int[] tempEventList = MLFuncs.getItemsAtIndices(eventList, MLFuncs.range(first, last - 1));
			Matrix tempDesignScores = MLFuncs.getRows(designScores, MLFuncs.range(groupIdx * numConditions, ((groupIdx + 1) * numConditions) - 1));
			tempDesignScores = MLFuncs.getRows(tempDesignScores, MLFuncs.subtract(tempEventList, 1));
			
			newDesignScores = MLFuncs.append(newDesignScores, tempDesignScores);
		}
	}
	//added for PET
	public ExpandDesignScores(Matrix designLV, int [] numSubjectLst, int[]numConditionList, int numGroups) {
		int numCol = designLV.getColumnDimension();
		int k, n;
		for(int i = 1; i <= numGroups; i++) {
			k = numConditionList[i-1];
			n = numSubjectLst[i-1];
			int start = (i - 1) * k;
			int end =   (i - 1) * k + k - 1;
			int[] range = MLFuncs.range(start, end);
			Matrix tempDesignScores = MLFuncs.reshape(MLFuncs.getRows(designLV, range), 1, numCol * k);
			tempDesignScores = MLFuncs.replicateRows(tempDesignScores, n);
			tempDesignScores = MLFuncs.reshape(tempDesignScores, n * k, numCol);
			
			newDesignScores = MLFuncs.append(newDesignScores, tempDesignScores);
		}
		
	}
}
