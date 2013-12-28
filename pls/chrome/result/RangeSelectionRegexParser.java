package pls.chrome.result;

import java.util.ArrayList;
import java.util.TreeSet;

import pls.shared.GlobalVariablesFunctions;

public class RangeSelectionRegexParser {
	
	// The regular expressions used to for parsing the input fields related
	// to lag, lv and slice selection.
	private static final String NUMBER_REGEX = "\\d+";
	private static final String HYPHEN_RANGE_REGEX = "\\d+\\s*-\\s*\\d+";
	private static final String COMMA_RANGE_REGEX = "(\\d+\\s*,\\s*)+\\d+";
	private static final String MULTI_RANGE_REGEX = "(\\d+|\\d+\\s*-\\s*\\d+)(\\s*,\\s*(\\d+|\\d+\\s*-\\s*\\d+))+";
	
	/**
	 * Given a text, will attempt to parse it into a list of numbers.
	 * So, "4-6, 8" will return the integers {4,5,6,8}.
	 */
	public static ArrayList<Integer> parseString(String range, String varName, int min, int max) {
		TreeSet<Integer> numbers = new TreeSet<Integer>();
		
		// Checks if the required field was filled in first.
		if (range.equals("")) {
			GlobalVariablesFunctions.showErrorMessage("The input for " + varName + " cannot be empty.");
			return null;
			
		// Otherwise, checks if valid values were given as the range.
			
		// Checks if the given range is just a number.
		} else if (range.matches(NUMBER_REGEX)) {
			int typeNum = Integer.parseInt(range);
			if (checkNumber(typeNum, varName, min, max))
				numbers.add(typeNum);
			else
				return null;
			
		// Checks if the given range is of the form a-b. Spaces are
		// allowed before and after the hyphen (-).
		} else if (range.matches(HYPHEN_RANGE_REGEX)) {
			String[] newRange = range.split("-");
			int first = Integer.parseInt(newRange[0].trim());
			int last = Integer.parseInt(newRange[1].trim());
			
			for (int i = first; i <= last; ++i) {
				if (checkNumber(i, varName, min, max))
					numbers.add(i);
				else
					return null;
			}

		// Checks if the given range is of the form a, b, c. Spaces are
		// allowed before and after the commas.
		} else if (range.matches(COMMA_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				int typeNum = Integer.parseInt(newRange[i].trim());
				if (checkNumber(typeNum, varName, min, max))
					numbers.add(typeNum);
				else
					return null;
			}
		
		// Checks if the given range is a combination of the two cases
		// above.
		} else if (range.matches(MULTI_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				String rangeItem = newRange[i].trim();
				if (rangeItem.indexOf('-') != -1) {
					String[] miniRange = rangeItem.split("-");
					int first = Integer.parseInt(miniRange[0].trim());
					int last = Integer.parseInt(miniRange[1].trim());
					for (int j = first; j <= last; ++j) {
						if (checkNumber(j, varName, min, max))
							numbers.add(j);
						else
							return null;
					}
				} else {
					int typeNum = Integer.parseInt(rangeItem);
					if (checkNumber(typeNum, varName, min, max))
						numbers.add(typeNum);
					else
						return null;
				}
			}
			
		} else {
			GlobalVariablesFunctions.showErrorMessage("The input for " + varName + " is invalid.");
			return null;
		}
		
		return new ArrayList<Integer>(numbers);
	}
	
	private static boolean checkNumber(int num, String varName, int min, int max) {
		if (num < min) {
			GlobalVariablesFunctions.showErrorMessage("The input for " + varName + " contains values which are below the minimum of " + min + ".");
			return false;
		}
		else if (num > max) {
			GlobalVariablesFunctions.showErrorMessage("The input for " + varName + " contains values which are above the maximum of " + max + ".");
			return false;
		}
		
		return true;
	}
}
