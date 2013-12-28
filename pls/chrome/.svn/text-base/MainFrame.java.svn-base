package pls.chrome;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JSeparator;
import javax.swing.UIManager;
import javax.swing.border.Border;

import pls.chrome.analysis.AnalysisFrame;
import pls.chrome.analysis.NpairsAnalysisFrame;
import pls.chrome.analysis.PetAnalysisFrame;
import pls.chrome.result.LoadedVolumesDialog;
import pls.chrome.result.PlsResultLoader;
import pls.chrome.result.ResultFrame2;
import pls.chrome.sessionprofile.PetSessionProfileFrame;
import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.chrome.shared.BaseMenuBar;
import pls.othertools.niftiextractor.NiftiExtractorBatch;
import pls.othertools.rvptool.PredictionVsPCNumTool;
import pls.othertools.rvptool.ReproducibilityVsPredictionTool;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.PetResultFileFilter;
import pls.shared.ResultFileFilter;
import pls.shared.fMRIResultFileFilter;

@SuppressWarnings("serial")
public class MainFrame extends JFrame {
	private final String helpString = "This is the main window.  From here you can\n" + 
	"a) Create session files and datamats.\n" + 
	"b) Run a PLS or NPAIRS analysis.\n" + 
	"c) View the results of a PLS or NPAIRS analysis.";
	private final String imageURL = "/images/plslogo.gif";
	private boolean loggingEnabled = false;
	private String now;
	public PrintStream outputLogStream;
	private MainContent mContent;
	
	public MainFrame() throws Exception {
		String lookAndFeel = UIManager.getSystemLookAndFeelClassName();
     	try {
		    UIManager.setLookAndFeel(lookAndFeel);
		} catch(Exception e) {
			// whatever, do not even tell the user about this.
		} finally {
			JFrame.setDefaultLookAndFeelDecorated(true);
		}

        // Create and set up the window.
        setTitle("PLS and NPAIRS Analysis");
        //setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
        Container content = getContentPane();

        // Create the menu bar.
        MainMenuBar menuBar = new MainMenuBar(this);
        menuBar.setHelpData(helpString);
        setJMenuBar(menuBar);

        // Create the content of the main window
        mContent = new MainContent();
        content.add(mContent);

        // Causes the layout manager to resize the frame
        pack();
        
        // Centers the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setResizable(false);
        
        setVisible(true);
	}
	
	public void dispose() {
		super.dispose();
		// this thing erases any log files that are empty
		if (loggingIsEnabled()) {
			System.setErr(System.out);
			if (outputLogStream != null) {
				outputLogStream.flush();
				outputLogStream.close();
			}
			String s = System.getProperty("file.separator");
			boolean weNeedToKillMoreLogs = true;
			int logNum = 1;
			while(weNeedToKillMoreLogs) {
				//format of this string needs to be the same as logFiles in method: createErrorLog() in Main class
				File logFile = new File("error_logs" + s + now + "_plsnpairs_error_log" + logNum + ".txt");
				if(logFile.exists()) {
					try {
						BufferedReader br = new BufferedReader(new FileReader(logFile));
						if (br.readLine() == null) {
							br.close();
							if (logFile.delete())
								System.out.println("PLSNPAIRS error log" + logNum + " was erased, because there were no errors reported");
						}						
					} catch (FileNotFoundException e) {
						System.out.println("There was a FileNotFoundException when trying to erase empty log files");
						e.printStackTrace();
					} catch (IOException e) {
						System.out.println("There was an IOException when trying to erase empty log files");
						e.printStackTrace();
					}
				} else {
					weNeedToKillMoreLogs = false;
				}
				logNum++;
			}
		}
		
		System.exit(0);
	}
	
	public void setLoggingEnabled(boolean loggingEnabled) {
		this.loggingEnabled = loggingEnabled;
	}

	public boolean loggingIsEnabled() {
		return loggingEnabled;
	}

	public void setNow(String now) {
		this.now = now;
	}

	public String getNow() {
		return now;
	}

	/**
	 * Retrieve the "Show Results" button so that when we open a new result 
	 * viewer window when running the "BATCH" command via the command line
	 * we can disable this button.
	 * @return the "Show Results" button.
	 */
	public JButton getResultsButton(){
		return mContent.getResultsButton();
	}
	/**
	 * Helper to create the content.
	 */
	final class MainContent extends JPanel implements ActionListener  {
		
		JRadioButton[] analysisTypeRadioButtons;
		JRadioButton[] expTypeRadioButtons;
		
		JButton sessionProfileButton;
		JButton runPlsAnalysisButton;
		JButton showPlsResultsButton;
		
		JComboBox otherToolsComboBox;
		
		public MainContent() {
			setLayout(new BorderLayout());
			
			// Add the PLS logo
			java.net.URL imgURL = this.getClass().getResource(imageURL);
			JLabel logo = new JLabel(new ImageIcon(imgURL));
			add(logo, BorderLayout.NORTH);
			
			// Analysis type radio buttons
			final int numAnalysisTypes = 2;
			analysisTypeRadioButtons = new JRadioButton[numAnalysisTypes];
			// Experiment type radio buttons
	        final int numExperimentTypes = 4;
	        expTypeRadioButtons = new JRadioButton[numExperimentTypes];
	        
	        ButtonGroup expGroup = new ButtonGroup();
	        ButtonGroup anGroup = new ButtonGroup();
	        
	        analysisTypeRadioButtons[0] = new JRadioButton("PLS");
	        analysisTypeRadioButtons[1] = new JRadioButton("NPAIRS");
	        
	        expTypeRadioButtons[0] = new JRadioButton("Blocked fMRI");
	        expTypeRadioButtons[1] = new JRadioButton("Event related fMRI");
	        expTypeRadioButtons[2] = new JRadioButton("PET");
	        expTypeRadioButtons[3] = new JRadioButton("ERP");
	        
	        for (int i = 0; i < numAnalysisTypes; i++) {
	        	anGroup.add(analysisTypeRadioButtons[i]);
	        	analysisTypeRadioButtons[i].setActionCommand("analysis type");
	        	analysisTypeRadioButtons[i].addActionListener(this);
	        }
	        
	        for(int i = 0; i < numExperimentTypes; i++) {
	            expGroup.add(expTypeRadioButtons[i]);
	        }
	        
	        
	        // Select the first buttons by default.
	        analysisTypeRadioButtons[0].setSelected(true);
	        expTypeRadioButtons[0].setSelected(true);
	        
	        // Disable unimplemented features
	        expTypeRadioButtons[2].setEnabled(false);
	        expTypeRadioButtons[3].setEnabled(false);
	        
		    Border border = BorderFactory.createTitledBorder("Analysis type:");
			JPanel radioBox = new JPanel();
			radioBox.setBorder(border);
			radioBox.setLayout(new BoxLayout(radioBox, BoxLayout.X_AXIS));
			for(int i = 0; i < numAnalysisTypes; i++) {
				radioBox.add(analysisTypeRadioButtons[i]);
			}
			
			Border border2 = BorderFactory.createTitledBorder("Experiment type:");
			JPanel radioBox2 = new JPanel();
			radioBox2.setBorder(border2);
			radioBox2.setLayout(new BoxLayout(radioBox2, BoxLayout.X_AXIS));
			for (int i = 0; i < numExperimentTypes; i++) {
				radioBox2.add(expTypeRadioButtons[i]);
			}

			add(radioBox, BorderLayout.WEST);
			add(radioBox2, BorderLayout.EAST);
			
			JPanel buttonPane = new JPanel(new GridLayout(1, 3));
			
	        sessionProfileButton = new JButton("Session Profile");
	        sessionProfileButton.addActionListener(new ActionListener() { 
	        	public void actionPerformed(ActionEvent e) {
	        		JFrame sessionProfileFrame = null;
	        		if (analysisTypeRadioButtons[0].isSelected()) {
	        			// PLS Analysis selected
	        			String PLSMenuBarTitle = "Create new PLS session information";
	        			if(expTypeRadioButtons[0].isSelected()) {
	        				// blocked fMRI PLS
	        				sessionProfileFrame = new SessionProfileFrame(true, false, PLSMenuBarTitle);
	        			} else if(expTypeRadioButtons[1].isSelected()) {
	        				// event-related fMRI PLS
	        				sessionProfileFrame = new SessionProfileFrame(false, false, PLSMenuBarTitle);
	        			} else if(expTypeRadioButtons[2].isSelected()) {
	        				// PET PLS
	        				sessionProfileFrame = new PetSessionProfileFrame();
	        			}
	        		}
	        		else {
	        			// NPAIRS Analysis selected - exp. type must be fMRI
	        			String NPAIRSMenuBarTitle = "Create new NPAIRS session information";
	        			if (expTypeRadioButtons[0].isSelected()) {
	        				// blocked fMRI NPAIRS
	        				sessionProfileFrame = new SessionProfileFrame(true, true, NPAIRSMenuBarTitle);
	        			}
	        			else if (expTypeRadioButtons[1].isSelected()) {
	        				// event-related fMRI NPAIRS
	        				sessionProfileFrame = new SessionProfileFrame(false, true, NPAIRSMenuBarTitle);
	        			}
	        		}
	        		
	        		if (sessionProfileFrame != null) {
	        			
	        			// Disables the button to open this window after it has
	        			// been opened. We only want one session profile window
	        			// to be opened by one instance of this application.
		        		sessionProfileFrame.addWindowListener(new WindowAdapter() {
		        			
		        			public void windowOpened(WindowEvent e) {
		        				sessionProfileButton.setEnabled(false);
		        			}
		        			
		        			public void windowClosed(WindowEvent e) {
		        				sessionProfileButton.setEnabled(true);
		        				((JFrame)e.getSource()).removeWindowListener(this);
		        			}
		        			
		        			public void windowClosing(WindowEvent e) {
		        				windowClosed(e);
		        			}
		        		});
	        		}
	        	}
	        });
	        buttonPane.add(sessionProfileButton);
	        
	        runPlsAnalysisButton = new JButton("Run Analysis");
	        runPlsAnalysisButton.addActionListener(new ActionListener() {
	        	public void actionPerformed(ActionEvent e) {
	        		JFrame analysisFrame = null;
	        		if (analysisTypeRadioButtons[0].isSelected()) {
	        			// PLS analysis selected
	        			if(expTypeRadioButtons[0].isSelected()) {
	        				// blocked fMRI PLS
			        		analysisFrame = new AnalysisFrame(true);
		        		} else if(expTypeRadioButtons[1].isSelected()) {
		        			// event-related fMRI PLS
		        			analysisFrame = new AnalysisFrame(false);
		        		} else if(expTypeRadioButtons[2].isSelected()) {
		        			// PET PLS
		        			analysisFrame = new PetAnalysisFrame(true);
		        		}
	        		}
	        		else {
	        			// NPAIRS analysis selected: must be fMRI
	        			if(expTypeRadioButtons[0].isSelected()) {
	        				// blocked fMRI NPAIRS
			        		analysisFrame = new NpairsAnalysisFrame(true);
	        			} else if (expTypeRadioButtons[1].isSelected()) {
	        				// event-related fMRI NPAIRS
	        				analysisFrame = new NpairsAnalysisFrame(false);
	        			}
	        		}
	        		
	        		if (analysisFrame != null) {
	        			
	        			// Disables the button to open this window after it has
	        			// been opened. We only want one analysis window
	        			// to be opened by one instance of this application.
		        		analysisFrame.addWindowListener(new WindowAdapter() {
		        			
		        			public void windowOpened(WindowEvent e) {
		        				runPlsAnalysisButton.setEnabled(false);
		        			}
		        			
		        			public void windowClosed(WindowEvent e) {
		        				runPlsAnalysisButton.setEnabled(true);
		        				((JFrame)e.getSource()).removeWindowListener(this);
		        			}
		        			
		        			public void windowClosing(WindowEvent e) {
		        				windowClosed(e);
		        			}
		        		});
	        		}
	        		
	        	}});
	        buttonPane.add(runPlsAnalysisButton);
	        
	        showPlsResultsButton = new JButton("Show Results");
	        showPlsResultsButton.addActionListener(new ActionListener() {
	        	public void actionPerformed(ActionEvent e) {
	        		
	        		JFileChooser chooser = new JFileChooser(".");
	        		String fileExtension = null;
	        		String fileType = GlobalVariablesFunctions.PLS;
	        		
	        		if(analysisTypeRadioButtons[0].isSelected() && expTypeRadioButtons[0].isSelected()) {
	        			chooser.setFileFilter(new BfMRIResultFileFilter());
	        			fileExtension = BfMRIResultFileFilter.EXTENSION;
	        		} else if(expTypeRadioButtons[0].isSelected()) {
	        			chooser.setFileFilter(new NpairsfMRIResultFileFilter());
	        			fileExtension = NpairsfMRIResultFileFilter.EXTENSION;
	        			fileType = GlobalVariablesFunctions.NPAIRS;
	        		} else if(expTypeRadioButtons[1].isSelected()) {
	        			chooser.setFileFilter(new fMRIResultFileFilter());
	        			fileExtension = fMRIResultFileFilter.EXTENSION;
	        		} else if(expTypeRadioButtons[2].isSelected()) {
	        			chooser.setFileFilter(new PetResultFileFilter());
	        			fileExtension = PetResultFileFilter.EXTENSION;
	        		} else if(expTypeRadioButtons[3].isSelected()) {
	        			
	        		}
	        		//chooser.addChoosableFileFilter(new ResultFileFilter() );
					chooser.setFileFilter(new ResultFileFilter() );
					chooser.setPreferredSize(new Dimension(680, 480));
					chooser.setMultiSelectionEnabled(true);
					
	        		int option = chooser.showDialog(MainContent.this, "Select Results File");
	        		if(option == JFileChooser.APPROVE_OPTION) {
	        			File[] files = chooser.getSelectedFiles();
	        			List<String> filePaths = new LinkedList<String>();
	        			String lastPath;
	        			
	        			for(File file : files){
	        				filePaths.add(file.getAbsolutePath());
	        			}
	        			
	        			lastPath = ((LinkedList<String>) filePaths).getLast();
	        			//String fileName = chooser.getSelectedFile().getAbsolutePath();

						//Sets the directory to search for the next result files to the
						//one which we just used.
						LoadedVolumesDialog.setLastPath(PlsResultLoader.getPrefix(lastPath));

		        		ResultFrame2 results = new ResultFrame2(fileType, 
		        												filePaths);
		        		
		        		// Disables the button to open this window after it has
	        			// been opened. We only want one results window to be
	        			// opened by one instance of this application.
		        		results.addWindowListener(new WindowAdapter() {
		        			
		        			public void windowOpened(WindowEvent e) {
		        				showPlsResultsButton.setEnabled(false);
		        			}
		        			
		        			public void windowClosed(WindowEvent e) {
		        				showPlsResultsButton.setEnabled(true);
		        				((ResultFrame2)e.getSource()).removeWindowListener(this);
		        			}
		        			public void windowClosing(WindowEvent e) {
		        				windowClosed(e);
		        			}
		        			
		        		});
		        	}
	        	}
	        });
	        buttonPane.add(showPlsResultsButton);
	        JPanel moreButtons = new JPanel();
	        moreButtons.setLayout(new BoxLayout(moreButtons, BoxLayout.Y_AXIS));
	        moreButtons.add(buttonPane);
	        moreButtons.add(new JSeparator() );
	        
	        JPanel otherToolsPanel = new JPanel();
	        otherToolsPanel.setLayout(new BoxLayout(otherToolsPanel, BoxLayout.X_AXIS));
	        otherToolsPanel.add(new JLabel(" Other tools: "));
	        otherToolsComboBox = new JComboBox();
	        otherToolsComboBox.addItem("Reproducibility vs Prediction Plot Tool (NPAIRS)");
			otherToolsComboBox.addItem("Prediction vs PC # Plot Tool (NPAIRS)");
			otherToolsComboBox.addItem("Nifti image extractor");
	       // otherToolsComboBox.addItem("Other things as well.");
	        otherToolsPanel.add(otherToolsComboBox);
	        JButton otherToolsButton = new JButton("Go");
	        otherToolsButton.setActionCommand("other tools");
	        otherToolsButton.addActionListener(this);
	        otherToolsPanel.add(otherToolsButton);
	        moreButtons.add(otherToolsPanel);
	        
	        
			
//	        JButton npairsResultsButton = new JButton("Show NPAIRS-J Results");
//	        npairsResultsButton.addActionListener(new ActionListener() { 
//	        	public void actionPerformed(ActionEvent e) {
//	        		JFileChooser chooser = new JFileChooser(".");
//					chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
//					chooser.setMultiSelectionEnabled(true);
//	        		int option = chooser.showDialog(MainContent.this, "Select Results Directories");
//	        		if(option == JFileChooser.APPROVE_OPTION) {
//	        			File[] directories = chooser.getSelectedFiles();
//	        			String[] directoryNames = new String[directories.length];
//	        			for(int i = 0; i < directories.length; i++) {
//	        				directoryNames[i] = directories[i].getAbsolutePath();
//	        			}
//		        		new NpairsResultFrame(directoryNames);
//	        		}
//	        	}});
//	        buttonPane.add(npairsResultsButton);
	        add(moreButtons, BorderLayout.SOUTH);
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			String action = e.getActionCommand();
			if("other tools".equals(action)) {
				switch (otherToolsComboBox.getSelectedIndex()) {
				case 0:
					new ReproducibilityVsPredictionTool();
					break;
				case 1:
					new PredictionVsPCNumTool();
					break;
				case 2:
					new NiftiExtractorBatch().setVisible(true);
					break;
				default:
				}
			} else if ("analysis type".equals(action)) {
		        // If analysis type == "NPAIRS", can do Blocked (or Event-related when enabled) data analysis
				if (e.getSource() == analysisTypeRadioButtons[1]) {
					expTypeRadioButtons[0].setSelected(true);
					expTypeRadioButtons[1].setEnabled(true);
//					expTypeRadioButtons[1].setEnabled(false);
	    			expTypeRadioButtons[2].setEnabled(false);
				} 
				else {
					expTypeRadioButtons[1].setEnabled(true);
	        		//expTypeRadioButtons[2].setEnabled(true);
				}
			}
		}
		
		public JButton getResultsButton(){
			return showPlsResultsButton;
		}
	}
	@SuppressWarnings("serial")
	final class MainMenuBar extends BaseMenuBar {
		
		public MainMenuBar(JFrame frame) {
			
			super(frame);

	        JMenu settings = new JMenu("Settings");
	        settings.setMnemonic('S');
	        settings.getAccessibleContext().setAccessibleDescription("Configuration settings");
	        add(settings, 1);

	        ///////Disabled for now///////
	        settings.setEnabled(false);
	        //////////////////////////////
	        
	        JMenu matrixLibrary = new JMenu("Matrix Library");
	        	        
	        matrixLibrary.setMnemonic('M');
	        matrixLibrary.getAccessibleContext().setAccessibleDescription("Configuration settings");
	        settings.add(matrixLibrary);
	        
	        ButtonGroup matrixLibraries = new ButtonGroup();
	        for(int i = 0; i < GlobalVariablesFunctions.matrixLibraries.length; i++) {
	        	JRadioButtonMenuItem mat = new JRadioButtonMenuItem(GlobalVariablesFunctions.matrixLibraries[i]);
	        	mat.setActionCommand(GlobalVariablesFunctions.matrixLibraries[i]);
	        	mat.addActionListener(new SaveMatrixListener(frame, matrixLibraries));
		        mat.setSelected(GlobalVariablesFunctions.matrixLibraries[i].equals(
		        		GlobalVariablesFunctions.matrixLibrary));
		       //  Note: Matlab doesn't work in a timely way for 
		        // reasonably large data. Also, matlib choices are only valid for Npairs, not PLS
		        mat.setEnabled(true);
	        	matrixLibraries.add(mat);
		        matrixLibrary.add(mat);
	        }
	        
	        JMenu readme = new JMenu("Read me");
	        readme.setForeground(new java.awt.Color(238, 42, 18));
	        readme.addMouseListener(new MouseAdapter(){
	        	
	        	@Override
	        	public void mousePressed(MouseEvent e){
	        		JOptionPane.showMessageDialog(MainMenuBar.this,
	        				"Please note: mm voxel location values " +
	        				"in Results Viewer and extracted\nNifti " +
	        				"volumes will be incorrect if input data " +
	        				"is not in RAS orientation.");
	        	}
	        });
	        add(readme,3);
		}

		final class SaveMatrixListener implements ActionListener {
			
			private ButtonGroup matrixLibraries = null;
			
			public SaveMatrixListener(JFrame frame, ButtonGroup matrixLibraries) {
				this.matrixLibraries = matrixLibraries;
			}
			public void actionPerformed(ActionEvent e) {
				GlobalVariablesFunctions.matrixLibrary = matrixLibraries.getSelection().getActionCommand();
			}
		}
	}
}

// This abstract class extends the WindowListener interface and is mainly
// used to handle the event of windows being opened and closed.
//abstract class NewWindowListener implements WindowListener {
//	
//	public void windowActivated(WindowEvent e) {
//	}
//    
//	public void windowClosing(WindowEvent e) {
//		windowClosed(e);
//	}
//	
//    public void windowDeactivated(WindowEvent e) {
//    }
//    
//    public void windowDeiconified(WindowEvent e) {
//    }
//    
//    public void windowIconified(WindowEvent e) {
//    }
//}