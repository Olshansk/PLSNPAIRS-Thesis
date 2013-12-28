package pls.chrome.result.controller.observer.filters;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class SliceFiltersEvent extends Event {
	public void visit(Observer o) {
		if (o instanceof FiltersObserver) {
			((FiltersObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
