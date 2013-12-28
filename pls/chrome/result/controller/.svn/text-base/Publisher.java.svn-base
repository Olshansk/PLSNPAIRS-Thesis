package pls.chrome.result.controller;

import java.util.HashSet;

import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;

public class Publisher {
	private HashSet<Observer> mObservers = new HashSet<Observer>();
	
	public void registerObserver(Observer o) {
		mObservers.add(o);
	}
	
	public synchronized void unregisterObserver(Observer o) {
		mObservers.remove(o);
	}
	
	public void publishEvent(Event e) {
		for (Observer o : mObservers) {
			e.visit(o);
		}
	}
	
	public void dispose() {
		mObservers.clear();
	}
}
