package pls;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Wrapper {
	public static void main(String[] argv) {
		for (String s: argv) {
			System.out.println(s);
		}
		String variable = System.getenv("PLSNPAIRS_HEAPSIZE");
		
		String javaHome = System.getProperty("java.home");
		
		String path = "";
		try {
			path = new File(".").getCanonicalPath();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		System.out.println(path);
		int heapSize = 512;
		
		if (variable != null && variable.length() > 0) {
			try {
				heapSize = Integer.parseInt(variable);
			} catch (NumberFormatException nfex) {
				System.out.println("Error parsing heapsize, using default of 512");
			}
		}
		
		String command = javaHome + "/bin/java";
		String xmx = "-Xmx" + heapSize + "m";
		String cp = path + "/plsnpairs.jar";
		
		if (!new File(cp).exists()) {
			System.out.println("Error - " + cp + " does not exist! \nMake sure " + 
					"plsnpairs.jar is in current working directory.");
			System.exit(1);
		}
		ArrayList<String> params = new ArrayList<String>();
		params.add(command);
		params.add(xmx);
		params.add("-cp");
		params.add(cp);
		params.add("pls.PlsMain");
		for(String s : argv) {
			params.add(s);
		}
		
//		ProcessBuilder pb = new ProcessBuilder(command, xmx, "-cp", cp, "pls.PlsMain");
		ProcessBuilder pb = new ProcessBuilder(params);

		try {
			System.out.println("Starting process!!!");
			Process p = pb.start();
		} catch (IOException ioex) {
			ioex.printStackTrace();
		}
		
	}
}
