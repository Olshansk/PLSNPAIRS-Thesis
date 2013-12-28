package pls.chrome.result.controller.observer.colourscale;

import pls.chrome.result.controller.observer.Observer;

public interface ColourScaleObserver extends Observer {
	public void notify(ColourScaleEvent e);
}
