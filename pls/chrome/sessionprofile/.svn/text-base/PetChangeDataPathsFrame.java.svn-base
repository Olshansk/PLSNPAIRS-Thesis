package pls.chrome.sessionprofile;

import pls.sessionprofile.PetRunInformation;
import pls.sessionprofile.PetSubjectInformation;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import java.io.File;

import nl.jj.swingx.gui.modal.JModalFrame;

public class PetChangeDataPathsFrame extends JModalFrame {
	
	public PetChangeDataPathsFrame(PetSessionProfileFrame sessionFrame) {
		super("Change Data Paths");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
		
		if(sessionFrame.subjectInfo.isEmpty()) {
			JOptionPane.showMessageDialog(null, "You must have a subject to be able to change the data paths.", "Error", JOptionPane.ERROR_MESSAGE);
			dispose();
			return;
		}
		
		final JTextField pathField = new JTextField();
		pathField.setText(getCommonPath(sessionFrame.subjectInfo));
		pathField.setColumns(30);
		
		JButton browseButton = new JButton("Browse", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		browseButton.setIconTextGap(15);
	    
	    browseButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(pathField.getText());
				chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
				int option = chooser.showDialog(PetChangeDataPathsFrame.this, "Save results to");
				if(option == JFileChooser.APPROVE_OPTION) {
					pathField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	    JPanel browsePane = new JPanel(new GridLayout(1, 0, 10, 10));
	    browsePane.add(pathField);
	    browsePane.add(browseButton);
	    
		
		JButton storeButton = new JButton("Change Data Paths", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Save16.gif")));
		storeButton.setIconTextGap(15);
	    
		storeButton.addActionListener(new PetStorePathsListener(this, sessionFrame, pathField));

	    JPanel containerPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    containerPane.add(browsePane);
	    containerPane.add(storeButton);
	    
	    JPanel mainPane = new JPanel();
	    mainPane.add(containerPane);
	    
		add(mainPane);

        // Display the window
        pack();
        
        // Position the frame on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = (int)(screen.getWidth() - getWidth()) / 2;
	    int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setResizable(false);
        
        setVisible(true);
	}
	
	public static String getCommonPath(Vector<PetSubjectInformation> subjectInfo) {
		return new File(subjectInfo.get(0).dataDirectory).getParent();
	}
}

final class PetStorePathsListener implements ActionListener {

	private JFrame parent = null;
	private PetSessionProfileFrame sessionFrame = null;
	private JTextField pathField = new JTextField();
	
	public PetStorePathsListener(JFrame parent, PetSessionProfileFrame sessionFrame, JTextField pathField) {
		this.parent = parent;
		this.sessionFrame = sessionFrame;
		this.pathField = pathField;
	}
	public void actionPerformed(ActionEvent e) {
		File newPath = new File(pathField.getText());
		for(PetSubjectInformation subject : sessionFrame.subjectInfo) {
			String[] pathArray = subject.dataDirectory.split("/");
			pathArray = pathArray[pathArray.length - 1].split("\\\\");
			subject.dataDirectory = newPath + File.separator + pathArray[pathArray.length - 1];
		}
		sessionFrame.BrainPanelLoadImage(sessionFrame.subjectInfo);
		parent.dispose();
	}
}
