package pls.test;

public class EnumTest {
	
	private enum SplitType { SESSION, RUN };
	private static int count = 0;
	
	public EnumTest(String s) {
		
		try {
			SplitType sType = SplitType.valueOf(s.toUpperCase());
			System.out.println("Testing " + s + "(count = " + count + ")");
			++count;
		
		switch(sType) {
		case SESSION: 
			System.out.println("Session case! (count = " + count + ")" );
			++count;
			break;
		case RUN: System.out.println("Run case! (count = " + count + ")");
			++count;
			break;
		default: 
			System.out.println("Default... (count = " + count + ")");
			++count;
		}
		} catch (IllegalArgumentException e) {
			System.out.println(s + " is not a valid type!");
		}
		
	}
	
	public static void main (String[] args) {
		for (String a : args) {
			EnumTest eTest = new EnumTest(a);
		}
	}

}
