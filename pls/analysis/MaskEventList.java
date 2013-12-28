package pls.analysis;

import pls.shared.MLFuncs;

public class MaskEventList {
	
	protected int eventLength = 0;
	
	protected int[] eventList = null;
	
	protected int[] mask = null;
	
	public MaskEventList(int[] eventList, int[] conditionSelection) {
		this.eventList = eventList;
		int[] deselectedCond = MLFuncs.find(conditionSelection, 0);
		if(conditionSelection.length == 0 || deselectedCond.length == 0) {
			eventLength = eventList.length;
			mask = MLFuncs.range(0, eventLength - 1);
			return;
		}
		int[] matchedCond = deselectedCond.clone();
		int[] tempEventList = eventList.clone();
		mask = MLFuncs.ones(tempEventList.length);
		
		for(int j = 0; j < deselectedCond.length; j++) {
			mask = MLFuncs.setValues(mask, MLFuncs.find(tempEventList, deselectedCond[j] + 1), 0);
			eventList = MLFuncs.setValues(eventList, MLFuncs.find(tempEventList, deselectedCond[j] + 1), 0);
			
			for(int k = 0; k < eventList.length; k++) {
				if(eventList[k] > matchedCond[j] + 1) {
					eventList[k]--;
				}
			}
			matchedCond = MLFuncs.subtract(matchedCond, 1);
		}
		
		mask = MLFuncs.find(mask, 1);
		this.eventList = MLFuncs.getItemsAtIndices(eventList, mask);
		eventLength = eventList.length;
	}
}
