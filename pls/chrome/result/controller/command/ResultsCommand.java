package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * This abstract class represents commands used in the Results Viewer.  It
 * requires that a repository be passed in.
 */
public abstract class ResultsCommand extends Command {
	protected GeneralRepository mRepository;
	
	public ResultsCommand(GeneralRepository repository) {
		mRepository = repository;
	}
}
