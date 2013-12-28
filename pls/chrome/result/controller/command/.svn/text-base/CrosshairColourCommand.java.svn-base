package pls.chrome.result.controller.command;

import java.awt.Color;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for setting the colour of the cross hairs in the single pane
 * brain viewer and the three pane brain viewer.
 */
public class CrosshairColourCommand extends ResultsCommand {
	private Color mColour;
	private Color mOldColour;

	public CrosshairColourCommand(GeneralRepository repository, Color colour) {
		super(repository);
		
		mOldColour = repository.getImagePropertiesModel().getCrosshairColour();
		mColour = colour;
		
		mCommandLabel = "set crosshair colour";
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setCrosshairColour(mColour);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.getImagePropertiesModel().setCrosshairColour(mOldColour);
		
		return true;
	}

}
