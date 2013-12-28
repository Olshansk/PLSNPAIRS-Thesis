package pls.chrome.result.model;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.selection.SelectionEvent;

/**
 * 
 */
public class SelectionModel {
	private int[] mSelectedVoxel = null;
	
	private Publisher mPublisher = null;
	
	protected void setPublisher(Publisher pub) {
		mPublisher = pub;
	}
	
	public void selectVoxel(int x, int y, int z, int lag) {
		mSelectedVoxel = new int[]{x, y, z, lag};
		
		if (mPublisher != null) {
			mPublisher.publishEvent(new SelectionEvent() );
		}
	}
	
	/**
	 * Returns an int array representing the currently selected voxel in the
	 * format {x, y, z, lag}.
	 */
	public int[] getSelectedVoxel() {
		return mSelectedVoxel;
	}
}
