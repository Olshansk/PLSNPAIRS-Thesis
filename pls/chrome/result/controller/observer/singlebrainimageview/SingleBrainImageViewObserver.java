package pls.chrome.result.controller.observer.singlebrainimageview;


public interface SingleBrainImageViewObserver {
	public void notify(BrainViewEvent e);
	
	public void notify(RotationEvent e);
	
	public void notify(ZoomEvent e);
}
