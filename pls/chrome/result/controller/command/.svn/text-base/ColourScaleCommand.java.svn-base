package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * This command sets the colour scale for a particular LV.
 */
public class ColourScaleCommand extends SelectionDependentCommand {
	private double mMax;
	private double mMin;
	private double mThreshold;
	
	private double mOrigMax;
	private double mOrigMin;
	private double mOrigThreshold;
	
	public ColourScaleCommand(GeneralRepository repository, double max, double min, double threshold) {
		super(repository);
		
		mMax = max;
		mMin = min;
		mThreshold = threshold;
		
		double[] oldColour = mRepository.getGeneral().getBrainData().getColourScaleModel().getColourScale();
		mOrigMax = oldColour[0];
		mOrigMin = oldColour[1];
		mOrigThreshold = oldColour[2];
		
		mCommandLabel = "Set colour scale";
	}

	@Override
	protected boolean postSelectionDo() {
		mRepository.getGeneral().getBrainData().getColourScaleModel().setColourScale(mMax, mMin, mThreshold);
		
		return true;
	}

	@Override
	protected boolean postSelectionUndo() {
		mRepository.getGeneral().getBrainData().getColourScaleModel().setColourScale(mOrigMax, mOrigMin, mOrigThreshold);
		
		return true;
	}
}
