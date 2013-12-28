package pls.chrome.result.controller.observer.colourscale;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class ColourScaleEvent extends Event {
	public void visit(Observer o) {
		if (o instanceof ColourScaleObserver) {
			((ColourScaleObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
