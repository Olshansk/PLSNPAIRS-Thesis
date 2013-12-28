package pls.othertools.rvptool;

import java.util.ArrayList;
import java.util.Collections;

/**
 * This class represents a datastructure for holding information about a
 * single 'curve'. This group/curve is then plotted in the Prediction vs PC #
 * plot.
 */
public class CurveGroup {
	private String mLabel;
	private String mGeneralFilename;
	private int mCUnit;
	private String mNumbers;

	public CurveGroup(String label, String filename, int cUnit, String numbers) {
		mLabel = label;
		mGeneralFilename = filename;
		mCUnit = cUnit;
		mNumbers = numbers;
	}
	
	public void setLabel(String label) {
		mLabel = label;
	}
	
	public String getLabel() {
		return mLabel;
	}
	
	public void setFilename(String filename) {
		mGeneralFilename = filename;
	}
	
	public String getFilename() {
		return mGeneralFilename;
	}

	/**
	 * Set the cv num / session / subject / split object that this group is
	 * belongs to.
	 * @param cUnit the unit of interest that this group is tied to.
	 */
	public void setCurveUnit(int cUnit) {
		mCUnit = cUnit;
	}
	
	public int getCurveUnit() {
		return mCUnit;
	}
	
	public void setNumbers(String numbers) {
		mNumbers = numbers;
	}
	
	public String getNumbers() {
		return mNumbers;
	}
	
	public ArrayList<String> getParsedNumbers() {
		return parseNumbers(mNumbers);
	}
	
	private ArrayList<String> parseNumbers(String numbers) {
		final String rangeWithStep = "\\d+\\s*-\\s*\\d+\\s*s\\s*\\d+";
		final String range = "\\d+\\s*-\\s*\\d+";
		final String singleNumber = "\\d+";
		
		ArrayList<String> returnVal = new ArrayList<String>();
		
		String[] pieces = numbers.split(",");
		
		for (String s : pieces) {
			s = s.trim();
			
			if (s.matches(rangeWithStep) ) {
				int min = Integer.parseInt(s.substring(0, s.indexOf("-") ).trim() );
				int max = Integer.parseInt(s.substring(s.indexOf("-") + 1, s.indexOf("s") ).trim() );
				int step = Integer.parseInt(s.substring(s.indexOf("s") + 1) );
				
				for (int i = min; i <= max; i += step) {
					returnVal.add(padStringWithZeros(Integer.toString(i) ) );
				}
			} else if (s.matches(range) ) {
				int min = Integer.parseInt(s.substring(0, s.indexOf("-") ).trim() );
				int max = Integer.parseInt(s.substring(s.indexOf("-") + 1).trim() );
				
				for (int i = min; i <= max; ++i) {
					returnVal.add(padStringWithZeros(Integer.toString(i) ));
				}
			} else if (s.matches(singleNumber) ) {
				returnVal.add(padStringWithZeros(s) );
			}
		}
		
		Collections.sort(returnVal);
		
		return returnVal;
	}
	
	private String padStringWithZeros(String input) {
		String output = input;
		
		while(output.length() < 3) {
			output = "0" + output;
		}
		
		return output;
	}
}
