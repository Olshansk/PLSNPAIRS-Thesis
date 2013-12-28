package pls.chrome.result.controller.observer.selection;

import pls.chrome.result.controller.observer.Observer;

public interface SelectionObserver extends Observer {
	public void notify(SelectedDataTypeChangedEvent e);
	
	public void notify(SelectedLvChangedEvent e);
	
	public void notify(SelectedVolumeChangedEvent e);
	
	public void notify(SelectionEvent e);
}
