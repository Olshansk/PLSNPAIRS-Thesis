package pls.chrome.result.controller.command;

public abstract class Command {
	protected String mCommandLabel = "Command";
	protected boolean mUndoable = true;
	protected boolean mInvalidates = false;
	
	public abstract boolean execDo();
	
	public abstract boolean execUndo();
	
	/**
	 * Returns the Command Label which shows up in the edit menu.
	 * @return
	 */
	public String getCommandLabel() {
		return mCommandLabel;
	}
	
	/**
	 * Returns whether this command can be undone and thus whether it should
	 * be placed on the undo stack.
	 * @return
	 */
	public boolean isUndoable() {
		return mUndoable;
	}
	
	/**
	 * Returns whether this command invalidates previous commands (i.e. if the
	 * command changes the state so drastically that we cannot undo previous
	 * commands.)
	 * @return
	 */
	public boolean invalidates() {
		return mInvalidates;
	}
}
