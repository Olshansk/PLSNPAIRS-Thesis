package pls.chrome.result.model;

import java.util.ArrayList;
import java.util.HashMap;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.Observer;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;

/**
 * A model containing the data used by the main LV Viewer.  Is static because
 * there is presumably only need for one.
 */
public class ViewModel {
	private Publisher mPublisher = new Publisher();
//	private HashMap<String, String> mLvStrings = new HashMap<String, String>();
	private HashMap<String, ArrayList<Integer>> mLvs = new HashMap<String, ArrayList<Integer>>();
	private String mSelectedDataType = null;
	
	protected void setPublisher(Publisher publisher) {
		mPublisher = publisher;
	}
	
	public void registerObserver(Observer o) {
		mPublisher.registerObserver(o);
	}
	
	public void setViewedLvs(ArrayList<Integer> viewedLvs) {
		setViewedLvs(mSelectedDataType, viewedLvs);
	}

	public void setViewedLvs(String dataType, ArrayList<Integer> viewedLvs) {
		mLvs.put(dataType, viewedLvs);
		
		mPublisher.publishEvent(new ViewedLvsEvent() );
	}
	
	public ArrayList<Integer> getViewedLvs(String dataType) {
		return mLvs.get(dataType);
	}
	
	public void setSelectedDataType(String dataType) {
		mSelectedDataType = dataType;
	}
}