package pls.chrome.result.controller.observer.brainimageproperties;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class UseCrosshairEvent extends Event {
	@Override
	public void visit(Observer o) {
		if (o instanceof BrainImagePropertiesObserver) {
			((BrainImagePropertiesObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
