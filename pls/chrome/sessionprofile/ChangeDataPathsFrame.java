package pls.chrome.sessionprofile;


import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.Vector;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import nl.jj.swingx.gui.modal.JModalFrame;
import pls.sessionprofile.RunInformation;

@SuppressWarnings("serial")
public class ChangeDataPathsFrame extends JModalFrame {

	private JFileChooser browser;

	/**
	 * This is the result of clicking on Edit -> Change Data Paths in the
	 * session profile viewer.
	 * @param sessionFrame The main session profile frame??
	 */
	public ChangeDataPathsFrame(SessionProfileFrame sessionFrame) {
		super("Change Data Paths");
		final JTextField pathField;
		JButton browseButton;
		JButton storeButton;

		JPanel browsePane;
		JPanel containerPane;
		JPanel mainPane;

		Dimension screen;

		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);

		//Return since no session files have been loaded.
		if(sessionFrame.runInfo.isEmpty()) {
			JOptionPane.showMessageDialog(null, 
					"You must have a run to be able to change the data paths.",
					"Error", JOptionPane.ERROR_MESSAGE);
			dispose();
			return;
		}

		//Set up the pathfield.
		pathField = new JTextField();
		//pathField.setText(getCommonPath(sessionFrame.runInfo));
		pathField.setText(sessionFrame.getLastPath());
		pathField.setColumns(30);

		//Set up the browser dialog
		browser = new JFileChooser(sessionFrame.getLastPath());
		removeExtraButtons(browser.getComponents());
		browser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		browser.setMultiSelectionEnabled(false);
		browser.setDialogTitle("Select Data Path");
		browser.setApproveButtonText("Enter");

		browseButton = new JButton("Browse", 
				new ImageIcon(this.getClass()
				.getResource("/toolbarButtonGraphics/general/Open16.gif")));
		browseButton.setIconTextGap(15);
	    
	    browseButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				int option = browser.showOpenDialog(ChangeDataPathsFrame.this);
				if(option == JFileChooser.APPROVE_OPTION) {
					pathField.setText(
							browser.getSelectedFile().getAbsolutePath());
				}
			}
		});

		browsePane = new JPanel(new GridLayout(1, 0, 10, 10));
	    browsePane.add(pathField);
	    browsePane.add(browseButton);
	    
		storeButton = new JButton("Change Data Paths", 
				new ImageIcon(this.getClass()
				.getResource("/toolbarButtonGraphics/general/Save16.gif")));
		storeButton.setIconTextGap(15);
	    
		storeButton.addActionListener(
				new StorePathsListener(this, sessionFrame, pathField));

	    containerPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    containerPane.add(browsePane);
	    containerPane.add(storeButton);
	    
	    mainPane = new JPanel();
	    mainPane.add(containerPane);
	    
		add(mainPane);

        // Display the window
        pack();
        
        // Position the frame on the screen
        screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = (int)(screen.getWidth() - getWidth()) / 2;
	    int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setResizable(false);
        
        setVisible(true);
	}
	
	public static String getCommonPath(Vector<RunInformation> runInfo) {
		return new File(runInfo.get(0).dataDirectory).getParent();
	}

	/**
	 * Strip the browse dialog of its extraneous buttons. Since I don't know
	 * where in the container hierarchy these buttons are located in recurse
	 * on every container found.
	 * @param comp list of components found in the current container.
	 */
	private void removeExtraButtons(Component[] comp){

		for(int i = 0; i < comp.length; i++){
			if(comp[i] instanceof JButton){
				JButton removal = (JButton) comp[i];
				String text = removal.getText();
				if(text != null){
					if(text.equals("New Folder") || text.equals("Rename File")
							|| text.equals("Delete File")){
						removal.setVisible(false);
					}
				}
			}else if(comp[i] instanceof Container){
				Container c = (Container) comp[i];
				removeExtraButtons(c.getComponents());
			}
		}
	}
}

final class StorePathsListener implements ActionListener {

	private JFrame parent = null;
	private SessionProfileFrame sessionFrame = null;
	private JTextField pathField;// = new JTextField();
	
	public StorePathsListener(JFrame parent, 
			                  SessionProfileFrame sessionFrame,
							  JTextField pathField) {
		this.parent = parent;
		this.sessionFrame = sessionFrame;
		this.pathField = pathField;
	}

	public void actionPerformed(ActionEvent e) {
		//File newPath = new File(pathField.getText());

		sessionFrame.setLastPath(pathField.getText());
		for(RunInformation run : sessionFrame.runInfo) {
			String imageDir = new File(run.dataDirectory).getName();
			String parentDir = pathField.getText();
			run.dataDirectory = parentDir + File.separator + imageDir;
		}
		parent.dispose();
		
		if (!sessionFrame.runInfo.isEmpty()) {
			if(!sessionFrame.checkDataPaths(false)) return;
			String dir = sessionFrame.runInfo.get(0).dataDirectory;
			String fileName = sessionFrame.runInfo.get(0).dataFiles.split(" ")[0];
			sessionFrame.loadImage(dir, fileName);
		}
	}
}