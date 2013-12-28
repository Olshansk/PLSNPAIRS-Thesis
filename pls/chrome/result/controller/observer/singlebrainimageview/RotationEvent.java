package pls.chrome.result.controller.observer.singlebrainimageview;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class RotationEvent extends Event {
	public void visit(Observer o) {
		if (o instanceof SingleBrainImageViewObserver) {
			((SingleBrainImageViewObserver)o).notify(this);
		}
		else {
			super.visit(o);
		}
	}
}
