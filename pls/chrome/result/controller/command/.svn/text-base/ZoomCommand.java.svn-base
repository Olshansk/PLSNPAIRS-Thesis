package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command to "zoom" in and out of the images in the single pane
 * brain viewer.
 */
public class ZoomCommand extends ResultsCommand {
	private double mZoomScale;
	
	public ZoomCommand(GeneralRepository repository, double zoomScale) {
		super(repository);
		
		mZoomScale = zoomScale;
		
		// Zoom is not undoable
		mUndoable = false;
		mInvalidates = false;
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setZoom(mZoomScale);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// Zoom is not undoable because of how often it might occur
		return false;
	}

}
