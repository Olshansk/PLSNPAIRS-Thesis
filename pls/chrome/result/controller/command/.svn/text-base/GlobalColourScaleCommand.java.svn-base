package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * This command sets the global colour scale.
 */
public class GlobalColourScaleCommand extends ResultsCommand {
	private double mMax;
	private double mMin;
	private double mThreshold;
	
	private double mOrigMax;
	private double mOrigMin;
	private double mOrigThreshold;

	public GlobalColourScaleCommand(GeneralRepository repository, double max, double min, double threshold) {
		super(repository);
		
		mMax = max;
		mMin = min;
		mThreshold = threshold;
		
		double[] oldColour = mRepository.getGlobalColourScale();
		mOrigMax = oldColour[0];
		mOrigMin = oldColour[1];
		mOrigThreshold = oldColour[2];
		
		mCommandLabel = "Set global colour scale";
	}

	@Override
	public boolean execDo() {
		mRepository.setGlobalColourScale(mMax, mMin, mThreshold);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.setGlobalColourScale(mOrigMax, mOrigMin, mOrigThreshold);
		
		return true;
	}

}
