package pls.chrome.result.controller.observer.datachange;

import pls.chrome.result.controller.observer.Observer;

public interface DataChangeObserver extends Observer {
	public void notify(FlipVolumeEvent e);
	
	public void notify(LoadedVolumesEvent e);
	
	public void notify(InvertedLvEvent e);
}
