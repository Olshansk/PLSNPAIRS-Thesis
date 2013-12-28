package pls.chrome.result.controller.observer.selection;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class SelectedLvChangedEvent extends Event {
	public void visit(Observer o) {
		if (o instanceof SelectionObserver) {
			((SelectionObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
