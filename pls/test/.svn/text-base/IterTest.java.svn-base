package pls.test;

import java.io.IOException;
import java.util.Iterator;

import npairs.io.NpairsjIO;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

public class IterTest {

	public static void main(String[] args) {
		try {

			MLStructure sessStruct = (MLStructure)new NewMatFileReader(args[0]).
				getContent().get("session_info");
			MLStructure runStruct = (MLStructure)sessStruct.getField("run");
			System.out.println("Size run struct: " + runStruct.getSize());
			
			String dataPath = ((MLChar)runStruct.getField("data_path")).getString(0);
			System.out.println("Data path: " + dataPath);
			for (int j = 0; j < runStruct.getSize(); ++j) {
				MLCell dataFiles = (MLCell)runStruct.getField("data_files", j);
				System.out.println("Data files: ");
				for (MLArray m : dataFiles.cells()) {
					System.out.println(((MLChar)m).getString(0));
				}
				
				String onsetFieldName = "evt_onsets";
				MLCell onsets = (MLCell)runStruct.getField(onsetFieldName, j);
				int numConds = -1;
				try {
					numConds = onsets.getN();
				}
				catch (NullPointerException npe) {
					onsetFieldName = "blk_onsets";
					onsets = (MLCell)runStruct.getField(onsetFieldName, j);
					numConds = onsets.getN();
					String blkLengthFieldName = "blk_length";
					MLCell lengths = (MLCell)runStruct.getField("blk_length");
					System.out.println("Size blk length: " + lengths.getSize());
					
				}
				if (onsets.getM() > 1) {
					numConds = onsets.getM();
				}
				
				System.out.println("Onsets: ");
				for (int m = 0; m < onsets.getM(); ++m) {
					for (int k = 0; k < onsets.getN(); ++k) {
						MLDouble currCondOnsets = (MLDouble)onsets.get(m,k);
						int currCond = k + 1;
						if (onsets.getM() > 1) {
							currCond = m + 1;
						}				
						System.out.println("Onsets (condition " + currCond + "): ");
						NpairsjIO.print(currCondOnsets.getArray());
					}
				}
		
			}
//			MLCell dataFiles0 = (MLCell)runStruct.getField("data_files0");
//			System.out.println("Size dataFiles0 MLCell: " + dataFiles0.getM() + ", " + dataFiles0.getN());
			
		
			
		}
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
}
