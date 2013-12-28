package pls.test;

import static org.junit.Assert.assertTrue;

import java.io.File;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;


/**
 * Tests involving the creation of new session files and the generation
 * of their datamats are excluded since a solution to filling out all the
 * necessary form values would be too time consuming to realize. 
 * Also, again, due to time limitations these tests are not fully automated.
 * The tester still needs to interact with the 'save as' dialogs. Simply 
 * pressing enter on the dialogs should be enough.
 * 
 * Tests will currently fail on any other machine than mine because the
 * loaded session files have the path's to their images hard coded. I still
 * need a way to modify the session files so their paths are updated on the
 * fly during testing so this test suite works on any machine.
 */

public class SessionDatamatPrefixTest {
	SessionProfileDatamatPrefix spf;
	String prefix = System.getProperty("user.dir") + File.separator + 
	"pls/test/sessionDatamatPrefixTests/";
	
	public static void main(String args[]){
		String prefix = System.getProperty("user.dir") + File.separator + 
		"pls/test/sessionDatamatPrefixTests/";
		
		SessionProfileDatamatPrefix spf = new SessionProfileDatamatPrefix();
		
		spf.load(prefix + "test2/session1_BfMRIsession.mat");
		//spf.setDatamatPrefix("bad_prefix");
		spf.generateDatamats();
	}
	
	@Before
	public void setup(){
		spf = new SessionProfileDatamatPrefix();
	}
	
	@After
	public void tearDown(){
		spf.dispose();
		spf = null;
	}
	
	/**
	 * Load a session file and then change the prefix. Attempt to save the new 
	 * session file. A prompt should come up asking for the save location.
	 */
	@Test
	public void testLoadBadPrefixAndSave(){
		
		spf.load(prefix + "test1/session1_BfMRIsession.mat");
		spf.setDatamatPrefix("bad_prefix");
		spf.save();
		
		File saveLocation = new File(spf.getSessionFilename());
		
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.exists());
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.delete()); 
		//clear up
	}
	
	/**Load a session file with a bad prefix then generate the associated
	 * datamats. The session file saved with the datamats should have the
	 * new bad (different than loaded session file prefix) prefix of the
	 * datamats.
	 */
	@Test
	public void testLoadGenerate(){
		spf.load(prefix + "test2/session1_BfMRIsession.mat");
		spf.setDatamatPrefix("bad_prefix");
		spf.generateDatamats();
		
		File saveLocation = new File(spf.getSessionFilename());
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.exists());
		
		
		String datamatLocation = saveLocation.getAbsolutePath();
		datamatLocation = datamatLocation.replace("BfMRIsession.mat",
												"BfMRIdatamat.mat");
		File datamat = new File(datamatLocation);
		assertTrue(datamatLocation,datamat.exists());
		
		//cleanup
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.delete());
		assertTrue(datamatLocation,datamat.delete());
	}
	
	/**
	 * Load a session file 
	 */
	@Test
	public void testLoadSaveGenerate(){
		spf.load(prefix + "test3/session1_BfMRIsession.mat");
		spf.save();
		spf.generateDatamats();
		
		File saveLocation = new File(spf.getSessionFilename());
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.exists());
		
		String datamatLocation = saveLocation.getAbsolutePath();
		datamatLocation = datamatLocation.replace("BfMRIsession.mat",
												"BfMRIdatamat.mat");
		File datamat = new File(datamatLocation);
		assertTrue(datamatLocation,datamat.exists());
		
		//cleanup
		assertTrue(datamatLocation,datamat.delete());
	}
	
	/**
	 * Load a session file, save, change the prefix, then run a datamat 
	 * generation. 
	 */
	@Test
	public void testLoadSaveChangeGen(){
		spf.load(prefix + "test4/session1_BfMRIsession.mat");
		spf.save();
		
		//Loaded file should still be there after saving.
		File saveLocation = new File(spf.getSessionFilename());
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.exists());
		
		//Change the prefix and run a generation
		spf.setDatamatPrefix("nprefix");
		spf.generateDatamats();
				
		saveLocation = new File(spf.getSessionFilename());
		
		//Check that output matches the prefix.
		String shortName = saveLocation.getName();
		
		shortName = shortName.substring(0, shortName.indexOf("_BfMRI"));
		assertTrue(shortName,shortName.equals("nprefix"));
		
		//assert file was actually saved.
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.exists());
		
		//assert datamat was saved.
		String datamatLocation = saveLocation.getAbsolutePath();
		datamatLocation = datamatLocation.replace("BfMRIsession.mat", 
				"BfMRIdatamat.mat");
		File datamat = new File(datamatLocation);
		assertTrue(datamatLocation,datamat.exists());
		
		//cleanup
		assertTrue(saveLocation.getAbsolutePath(),saveLocation.delete());
		assertTrue(datamatLocation,datamat.delete());
	}
}	
