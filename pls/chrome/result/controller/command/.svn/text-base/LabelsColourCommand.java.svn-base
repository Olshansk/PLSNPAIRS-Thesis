package pls.chrome.result.controller.command;

import java.awt.Color;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for changing the colour of the labels in the single pane
 * brain viewer.
 */
public class LabelsColourCommand extends ResultsCommand {
	
	private Color mColour;
	private Color mOldColour;

	public LabelsColourCommand(GeneralRepository repository, Color colour) {
		super(repository);
		
		mColour = colour;
		mOldColour = mRepository.getImagePropertiesModel().getLabelsColour();
		
		mCommandLabel = "Set Labels Colour";
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setLabelsColour(mColour);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.getImagePropertiesModel().setLabelsColour(mOldColour);
		
		return true;
	}

}
