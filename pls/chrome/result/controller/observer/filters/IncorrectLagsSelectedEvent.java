package pls.chrome.result.controller.observer.filters;

import java.util.ArrayList;
import java.util.HashMap;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

/**
 * This event occurs when a new file is loaded and is discovered
 * to not contain a lag that the user specified that they would like to see.
 * It also happens when the user hits "PLOT" and the same conditions exist as
 * above. This is done through the control panel where one selects which lags
 * they want to filter out.
 */
public class IncorrectLagsSelectedEvent extends Event {
	private HashMap<String, ArrayList<ArrayList<Integer>>> warnings;

	public IncorrectLagsSelectedEvent(
			HashMap<String, ArrayList<ArrayList<Integer>>> warning){
		warnings = warning;
	}

	/**
	 * Return the warnings generated when attempting to apply a lag filter
	 * to a certain result model that doesn't have the specified lags.
	 * @return The warnings associatively mapped by
	 * filename->brainview->invalidlags.
	 */
	public HashMap<String, ArrayList<ArrayList<Integer>>> getWarnings(){
		return warnings;
	}

	@Override
	public void visit(Observer o) {
		if (o instanceof FiltersObserver) {
			((FiltersObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
