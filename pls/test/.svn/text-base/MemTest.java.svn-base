package pls.test;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Vector;

import javax.swing.JFileChooser;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

public class MemTest {
	public static void main(String[] argv) throws Exception {
		System.out.println("Beginning test.");
		
		HashMap<String, Integer> testsToRun = new HashMap<String, Integer>();
		
		for (int i = 0; i < argv.length; i = i + 2) {
			testsToRun.put(argv[i], Integer.parseInt(argv[i+1]) );
		}
		
		if (testsToRun.size() == 0) {
			// Select a file
			JFileChooser chooser = new JFileChooser(".");
			int result = chooser.showOpenDialog(null);
			
			if (result == JFileChooser.APPROVE_OPTION) {
				// Open that mat file
				String fileName = chooser.getSelectedFile().getAbsolutePath();
				testsToRun.put(fileName, 1);
			}
		}
		
		Long begin = System.nanoTime();
		
		for (String s : testsToRun.keySet() ) {
			int i = testsToRun.get(s);
			
			for (int j = 0; j < i; ++j) {
				runTestOnFile(s);
			}
		}
		
		Long totalTime = System.nanoTime() - begin;
		
		System.out.println("The test took " + totalTime / 1000000000.0 + " seconds.");
		System.out.println("Ending test.");
	}
	
	private static void runTestOnFile(String fileName) throws Exception {
//		System.out.println("Testing with " + fileName);
		
		// Figure out what type of file this is
		String resultType = fileName.substring(fileName.lastIndexOf('_'), fileName.lastIndexOf('.') );
		resultType = resultType.substring(0, resultType.indexOf("result") );
		
		Map<String, MLArray> resultInfo = null;
		
		try {
			resultInfo = new NewMatFileReader(fileName, new MatFileFilter()).getContent();
		} catch (Exception ex) {
			System.err.println("Result file " + fileName + " could not be loaded.");
		}
		
		String fileDir = new File(fileName).getParent();
		
		// Get the session profile names
		Vector<String[]> sessionProfileNames = new Vector<String[]>();
		MLCell sp = (MLCell) resultInfo.get("SessionProfiles");
		for (int i = 0; i < sp.getN(); i++) {
			String[] currGroupProfile = null;
			MLCell currSp = (MLCell) sp.get(i);
			for (int c = 0; c < currSp.getM(); c++) {
				String thisc = ((MLChar) currSp.get(c)).contentToString();
				int first = thisc.indexOf('\'') + 1;
				int last = thisc.lastIndexOf('\'');
				currGroupProfile = MLFuncs.append(currGroupProfile, thisc
						.substring(first, last));
			}
			
			sessionProfileNames.add(currGroupProfile);
		}
		
		// Load session profiles into memory
		ArrayList<ArrayList<Map<String, MLArray> > > sessionProfiles = new ArrayList<ArrayList<Map<String, MLArray> > >();
		
		for (String[] sessionNames : sessionProfileNames) {
			ArrayList<Map<String, MLArray> > currGroup = new ArrayList<Map<String, MLArray> >();
			
			for (String s : sessionNames) {
				
				File file = new File(s);
				String shortName = file.getName();
				
				boolean fileFound = file.exists();
				if (!fileFound && fileDir != null) {
		    		s = fileDir + File.separator + shortName;
					file = new File(s);
					fileFound = file.exists();
				}
				
				if (!fileFound) {
					System.err.println("Could not find " + shortName + ".");
					System.exit(0);
				}
		
//				System.out.println("Loading session file " + s);
				currGroup.add(new NewMatFileReader(s, new MatFileFilter()).getContent() );
			}
			
			sessionProfiles.add(currGroup);
		}
		resultInfo = null;
		
		ArrayList<ArrayList<double[][]> > datamats = new ArrayList<ArrayList<double[][]> >();
		
		// Load the datamat files found in the session profiles
		for (ArrayList<Map<String, MLArray> > currGroup : sessionProfiles) {
			
			ArrayList<double[][]> currDatamatGroup = new ArrayList<double[][]>();
			
			for (Map<String, MLArray> currProfile : currGroup) {
				MLStructure sessionInfo = (MLStructure) currProfile.get("session_info");
				
				String datamatName = ((MLChar)sessionInfo.getField("datamat_prefix")).getString(0);
				datamatName += resultType + "datamat.mat";
				
				String fullDatamatName = fileDir + File.separator + datamatName;
				
				File file = new File(fullDatamatName);
				boolean fileFound = file.exists();
				
				if (!fileFound) {
					System.err.println("Could not find " + datamatName + ".");
					System.exit(0);
				}
				
				String[] filter = {"st_datamat"};
				
//				System.out.println("Loading datamat file " + fullDatamatName);
//				currDatamatGroup.add(new NewMatFileReader(fullDatamatName, new MatFileFilter()).getContent() );
//				currDatamatGroup.add(((MLDouble)new NewMatFileReader(fullDatamatName, new MatFileFilter(filter)).getContent().get("st_datamat")).getArray());
				double[][] d = ((MLDouble)new NewMatFileReader(fullDatamatName, new MatFileFilter(filter)).getContent().get("st_datamat")).getArray();
			}
			
			datamats.add(currDatamatGroup);
		}
	}
}
