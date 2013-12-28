package pls.chrome.result.controller.observer.datachange;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class LoadedVolumesEvent extends Event {	
	public void visit(Observer o) {
		if (o instanceof DataChangeObserver) {
			((DataChangeObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
