package pls.chrome.result.controller;

import java.util.LinkedList;

import pls.chrome.result.controller.command.Command;
import pls.chrome.result.controller.observer.Event;

class CommandManager {
	protected int mThreshold = 32;
	
	private LinkedList<Command> undoStack = new LinkedList<Command>();
	private LinkedList<Command> redoStack = new LinkedList<Command>();
	
	protected Publisher publisher = new Publisher();
	
	/**
	 * Attempts to execute a command and adds it to the undo stack if the
	 * command executes successfully.
	 * @param command The command to execute.
	 */
	void executeCommand(Command command) {
		boolean success = command.execDo();

		// Only mess with the command stacks if the command was executed
		// successfully.
		if (success) {
			// If the command invalidates the command stacks then
			// clear the command stacks.
			if (command.invalidates() ) {
				undoStack.clear();
				redoStack.clear();
			}
			
			// If the command is undoable, it must be added to the undo
			// stack and the redo stack must be cleared.
			if (command.isUndoable() ) {
				undoStack.addLast(command);
				
				if (undoStack.size() > mThreshold) {
					undoStack.removeFirst();
				}
				
				redoStack.clear();
			}
			
			// Necessary so that the undo/redo menu gets an update
			// AFTER the command has successfully executed.
			publisher.publishEvent(new Event() );
		}
	}
	
	/**
	 * Returns whether there are undoable commands in the undo stack.
	 */
	boolean canUndo() {
		return !undoStack.isEmpty();
	}
	
	/**
	 * Attempts to undo the first command in the undo stack, and adds it to
	 * the redo stack if successful.
	 */
	void undoCommand() {
		if (!undoStack.isEmpty() ) {
			Command command = undoStack.removeLast();
			
			boolean success = command.execUndo();
			
			if (success) {
				redoStack.addLast(command);
			}
			
			// Necessary so that the undo/redo menu gets an update
			// AFTER the command has successfully executed.
			publisher.publishEvent(new Event() );
		}
	}
	
	/**
	 * Returns the next command that would be undone if we were to
	 * execute an undo call.
	 */
	Command getNextUndoCommand() {
		Command rtn = null;
		
		if (!undoStack.isEmpty() ) {
			rtn = undoStack.getLast();
		}
		
		return rtn;
	}
	
	/**
	 * Returns whether there are redoable commands in the undo stack.
	 */
	boolean canRedo() {
		return !redoStack.isEmpty();
	}
	
	/**
	 * Attempts to redo the first command in the redo stack, and adds it to
	 * the undo stack if successful.
	 */
	void redoCommand() {
		if (!redoStack.isEmpty() ) {
			Command command = redoStack.removeLast();
			
			boolean success = command.execDo();
			
			if (success) {
				undoStack.addLast(command);
			}
			
			// Necessary so that the undo/redo menu gets an update
			// AFTER the command has successfully executed.
			publisher.publishEvent(new Event() );
		}
	}
	
	/**
	 * Returns the next command that would be redone if we were to
	 * execute an redo call.
	 */
	 Command getNextRedoCommand() {
		Command rtn = null;
		
		if (!redoStack.isEmpty() ) {
			rtn = redoStack.getLast();
		}
		
		return rtn;
	}
}
