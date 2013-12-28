package pls.shared;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class StreamedProgressHelper {
	private ArrayList<Double> mStartTimes = new ArrayList<Double>();
	private ArrayList<Boolean> mSubTaskIndicator = new ArrayList<Boolean>();
	private ArrayList<String> mShortDescriptions = new ArrayList<String>();
	
	private ArrayList<BufferedWriter> mWriters = new ArrayList<BufferedWriter>();
	private ArrayList<Boolean> mNeedsClosing = new ArrayList<Boolean>();
	
	public StreamedProgressHelper() {
//		if (os.equals(System.out) ) {
//			mNeedsClosing = false;
//		}
//		
//		mWriters = new BufferedWriter(new PrintWriter(os) );
	}
	
	public void addStream(OutputStream os) {
		if (os.equals(System.out) ) {
			mNeedsClosing.add(false);
		} else {
			mNeedsClosing.add(true);
		}
		
		mWriters.add(new BufferedWriter(new PrintWriter(os) ) );
	}
	
	public void postMessage(String message) {
		try {
	//		mStartTimes.add((double)System.currentTimeMillis());
			if (mSubTaskIndicator.size() > 0) {
				
				// If there haven't been previous subtasks, we'll need a newline
				if (!mSubTaskIndicator.get(mSubTaskIndicator.size() - 1) ) {
					for(BufferedWriter writer : mWriters) {
						writer.newLine();
					}
				}
				
				mSubTaskIndicator.set(mSubTaskIndicator.size() - 1, true);
				
				String spaces = "";
				for (int i = 0; i < mSubTaskIndicator.size(); ++i) {
					spaces += "    ";
				}
				if (mSubTaskIndicator.size() % 2 == 0) {
					spaces += "# ";
				} else {
					spaces += "* ";
				}
				
				for(BufferedWriter writer : mWriters) {
					writer.write(spaces);
				}
			}
			
			for(BufferedWriter writer : mWriters) {
				writer.write(message);
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void startTask(String description) {
		startTask(description, null);
	}
	
	public void startTask(String longDescription, String shortDescription) {
		try {
			mStartTimes.add((double)System.currentTimeMillis());
			if (mSubTaskIndicator.size() > 0) {
				
				// If there haven't been previous subtasks, we'll need a newline
				if (!mSubTaskIndicator.get(mSubTaskIndicator.size() - 1) ) {
					for(BufferedWriter writer : mWriters) {
						writer.newLine();
					}
				}
				
				mSubTaskIndicator.set(mSubTaskIndicator.size() - 1, true);
				
				String spaces = "";
				for (int i = 0; i < mSubTaskIndicator.size(); ++i) {
					spaces += "    ";
				}
				if (mSubTaskIndicator.size() % 2 == 0) {
					spaces += "# ";
				} else {
					spaces += "* ";
				}
				
				for(BufferedWriter writer : mWriters) {
					writer.write(spaces);
				}
			}
			
			mSubTaskIndicator.add(false);
			mShortDescriptions.add(shortDescription);
			
			for(BufferedWriter writer : mWriters) {
				writer.write(longDescription);
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void endTask() {
		try {
			String shortDescription = null;
			if (mShortDescriptions.size() > 0 ) {
				shortDescription = mShortDescriptions.remove(mShortDescriptions.size() - 1);
			}
			
			if (mSubTaskIndicator.size() > 0 && mSubTaskIndicator.remove(mSubTaskIndicator.size() - 1) ) {
				for(BufferedWriter writer : mWriters) {
					for (int i = 0; i < mSubTaskIndicator.size(); ++i) {
						writer.write("    ");
					}
				}
				
				if (shortDescription == null) {
					shortDescription = "Task";
				}

				String timeString = formatTime(mStartTimes.remove(mStartTimes.size() - 1), System.currentTimeMillis());
				for(BufferedWriter writer : mWriters) {
					writer.write(shortDescription + " completed in " + timeString + " seconds.");
				}
			} else {
				String timeString = formatTime(mStartTimes.remove(mStartTimes.size() - 1), System.currentTimeMillis());
				for(BufferedWriter writer : mWriters) {
					writer.write("   [done] (" + timeString + " seconds)");
				}
			}
			
			for(BufferedWriter writer : mWriters) {
				writer.newLine();
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public void printError(String errorText) {
		try {
			for(BufferedWriter writer : mWriters) {
				writer.newLine();
				writer.write("------------------------------");
				writer.newLine();
				writer.write("An error has occurred: ");
				writer.newLine();
				writer.write("------------------------------");
				writer.newLine();
				if (errorText != null) { 
					writer.write(errorText);
				}
				writer.newLine();
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void complete() {
		for(int i = 0; i < mWriters.size(); ++i) {
			if (mNeedsClosing.get(i) ) {
				BufferedWriter writer = mWriters.get(i);
				try {
					writer.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	private String formatTime(double startTime, double endTime) {
		return new DecimalFormat("0.0").format((endTime - startTime) / 1000.0);
	}
}
