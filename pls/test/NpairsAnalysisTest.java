package pls.test;

import java.io.IOException;
import java.io.PipedOutputStream;

import javax.swing.JFileChooser;
import javax.swing.JFrame;

import pls.analysis.NpairsAnalysis;
import pls.analysis.ResultSaver;
import pls.shared.NpairsAnalysisSetupFileFilter;
import pls.shared.StreamedProgressDialog;
import pls.shared.StreamedProgressHelper;

import junit.framework.TestCase;

import npairs.Npairsj;
import npairs.NpairsjException;
import npairs.NpairsjSetupParams;
import npairs.io.NpairsDataLoader;

/**
 * Some test cases that were written to determine why out of memory exceptions
 * sometimes occur when running several analyses in a row.
 */
public class NpairsAnalysisTest extends TestCase {
	/**
	 * This test runs an analysis multiple times to see if the program
	 * runs out of memory.
	 */
	public void testRunningSeveralAnalyses() {
		final int numberOfRuns = 0;
		
		final String matlibType = "COLT";
		final String matlibTypeForInitFeatSel = "COLT";
		
		JFileChooser jfc = new JFileChooser();
		jfc.setFileFilter(new NpairsAnalysisSetupFileFilter() );
		int result = jfc.showOpenDialog(null);
		
		if (result == JFileChooser.APPROVE_OPTION) {
			String npairsSetupParamsMatFileName = jfc.getSelectedFile().getAbsolutePath();
			
			int numCompletedAnalyses = 0;
			try {
				for (numCompletedAnalyses = 0; numCompletedAnalyses < numberOfRuns; ++numCompletedAnalyses) {
					NpairsjSetupParams nsp = new NpairsjSetupParams(npairsSetupParamsMatFileName, false);
					NpairsDataLoader ndl = new NpairsDataLoader(nsp, matlibType, 
							matlibTypeForInitFeatSel, false);
					Npairsj npairsj = new Npairsj(ndl, nsp, matlibType);
					new ResultSaver(npairsj, npairsSetupParamsMatFileName, nsp.resultsFilePrefix);
				}
			}
			catch (OutOfMemoryError oom) {
				fail("Ran out of memory after completing " + numCompletedAnalyses + " analyses.");
			}
			catch (IOException io) {
				fail("An unexpected I/O exception occurred.");
			}
			catch (NpairsjException npj) {
				fail("An unexpected NPAIRS-J exception occurred.");
			}
			catch (Exception e) {
				fail("Some other weird exception occurred.");
			}
			
			assertEquals(numberOfRuns, numCompletedAnalyses);
		}
		else {
			fail("The test was cancelled.");
		}
	}
	
	public void testRunningSeveralAnalysesWithProgressDialog() {
		final int numberOfRuns = 6;
		
		final String matlibType = "COLT";
		final String matlibTypeForInitFeatSel = "COLT";
		
		JFileChooser jfc = new JFileChooser();
		jfc.setFileFilter(new NpairsAnalysisSetupFileFilter() );
		int result = jfc.showOpenDialog(null);
		
		if (result == JFileChooser.APPROVE_OPTION) {
			String npairsSetupParamsMatFileName = jfc.getSelectedFile().getAbsolutePath();
			
			int numCompletedAnalyses = 0;
			
			for (numCompletedAnalyses = 0; numCompletedAnalyses < numberOfRuns; ++numCompletedAnalyses) {
				
				try {
					StreamedProgressDialog dialog = new StreamedProgressDialog(null, 100);
					PipedOutputStream pos = new PipedOutputStream();
					dialog.connectWriter(pos);
					StreamedProgressHelper helper = new StreamedProgressHelper();
					helper.addStream(pos);
					
					NpairsAnalysis worker = new NpairsAnalysis(npairsSetupParamsMatFileName, matlibType, 
							matlibTypeForInitFeatSel, true);
					
					dialog.worker = worker;
					
					worker.progress = helper;
					worker.run();
					
					dialog.dispose();
				} catch (IOException ioex) {
					ioex.printStackTrace();
				}
			}
		}
		else {
			fail("The test was cancelled.");
		}
	}
}
