package pls.chrome.result.controller.command;

import java.util.ArrayList;

import pls.chrome.result.model.GeneralRepository;

public class SliceFiltersCommand extends ResultsCommand {
	
	int mBrainView;
	private ArrayList<Integer> mLagNumbers;
	private ArrayList<Integer> mOldLagNumbers;
	ArrayList<Integer> mSliceNumbers;
	ArrayList<Integer> mOldSliceNumbers;
	int mNumRowsPerLag;
	int mOldNumRowsPerLag;
	boolean mDisplayAllLags;
	boolean mOldDisplayAllLags;

	public SliceFiltersCommand(GeneralRepository repository, int brainView, ArrayList<Integer> lagNumbers,
			ArrayList<Integer> sliceNumbers, int numRowsPerLag, boolean displayAll) {
		super(repository);

		mDisplayAllLags = displayAll;
		mOldDisplayAllLags  =  mRepository.getControlPanelModel().getAllLagsFlag(mBrainView);
		mBrainView = brainView;
		
		mLagNumbers = lagNumbers;
		mSliceNumbers = sliceNumbers;
		mNumRowsPerLag = numRowsPerLag;
		
		// Get the current filters so we can undo
		mOldLagNumbers = mRepository.getControlPanelModel().getLagNumbers(mBrainView);
		mOldSliceNumbers = mRepository.getControlPanelModel().getSliceNumbers(mBrainView);
		mOldNumRowsPerLag = mRepository.getControlPanelModel().getNumRowsPerLag(mBrainView);
		
		mCommandLabel = "Filter slices";
	}

	@Override
	public boolean execDo() {
		mRepository.getControlPanelModel().setFilters(
				mBrainView, mLagNumbers, mSliceNumbers, mNumRowsPerLag, mDisplayAllLags);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.getControlPanelModel().setFilters(
				mBrainView, mOldLagNumbers, mOldSliceNumbers, 
				mOldNumRowsPerLag, mOldDisplayAllLags);
		
		return true;
	}

}
