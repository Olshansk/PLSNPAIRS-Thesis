package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A class to rotate the images in the single pane brain viewer.
 */
public class RotateCommand extends ResultsCommand {
	private int mRotation;
	private int mOldRotation;
	
	/**
	 * The parameter clockwise determines if the rotation will be clockwise
	 * or counter clockwise.
	 * @param repository
	 * @param clockwise Rotates clockwise if true, counter clockwise otherwise.
	 */
	public RotateCommand(GeneralRepository repository, int rotation) {
		super(repository);
		
		mRotation = rotation;
		
		mOldRotation = mRepository.getImagePropertiesModel().getRotation();
		
		mCommandLabel = "Rotate Image";
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setRotation(mRotation);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.getImagePropertiesModel().setRotation(mOldRotation);
		
		return true;
	}

}
