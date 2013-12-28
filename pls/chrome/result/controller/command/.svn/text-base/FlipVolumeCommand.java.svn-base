package pls.chrome.result.controller.command;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;

public class FlipVolumeCommand extends SelectionDependentCommand {

	private boolean mFlipHorizontal;
	private boolean mFlipVertical;
	private int mBrainView;
	
	public FlipVolumeCommand(GeneralRepository repository, boolean flipHorizontal, boolean flipVertical, int brainView) {
		super(repository);
		
		String brainString = "Axial";
		
		if (brainView == BrainData.CORONAL) {
			brainString = "Coronal";
		}
		else if (brainView == BrainData.SAGITTAL) {
			brainString = "Sagittal";
		}
		mCommandLabel = "Flip Volume - " + brainString + " : " + (flipHorizontal ? "H" : "") + (flipVertical ? "V" : "");
		
		mFlipHorizontal = flipHorizontal;
		mFlipVertical = flipVertical;
		mBrainView = brainView;
	}

	@Override
	protected boolean postSelectionDo() {
		mRepository.getGeneral().flipVolume(mBrainView, mFlipHorizontal, mFlipVertical);
		
		return true;
	}

	@Override
	protected boolean postSelectionUndo() {
		return postSelectionDo();
	}

}
