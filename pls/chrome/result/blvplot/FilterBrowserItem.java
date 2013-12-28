package pls.chrome.result.blvplot;

@SuppressWarnings("serial")
public class FilterBrowserItem {
	private Object mObject;
	public boolean beingDisplayed = false;
	
	public FilterBrowserItem(Object o) {
		mObject = o;
	}
	
	public String toString() {
		return mObject.toString();
	}
}
