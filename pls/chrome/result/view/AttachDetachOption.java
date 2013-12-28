package pls.chrome.result.view;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import pls.chrome.result.ResultFrame;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlotManager;

public abstract class AttachDetachOption extends JPanel implements ActionListener, WindowListener {

	protected JButton mAttachDetachButton;
	//holds the absolute filenames of the matlab files used in the result
	//viewer.
	//The strings function as keys for retrieving the associated model. 
	//see getGeneral() in GeneralRepository.java
	public ArrayList<String> mResultFilePaths;
	
	public JFrame mDetachedFrame;
	protected PlotManager mManager;
	private String mTitle;
	
	public AttachDetachOption(GeneralRepository repository, String title) {
		mManager = repository.getPlotManager();
		mTitle = title;
		
		//JFrames are invisible by default.
		mDetachedFrame = new JFrame(title);
		mDetachedFrame.setSize(ResultFrame.DIMENSION);
		mDetachedFrame.addWindowListener(this);
		
		mAttachDetachButton = new JButton("Detach");
		mAttachDetachButton.addActionListener(this);
	}
	
	public String getTitle() {
		return mTitle;
	}
	
	public abstract void initialize();

	public void detachWindow() {
		mAttachDetachButton.setText("Attach");
		
		mDetachedFrame.add(this);
		mDetachedFrame.setVisible(true);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (mAttachDetachButton.getText().equals("Detach")) {
			mManager.detachPlot(this);
		} else {
			attachWindow();
		}
	}
	
	public void windowClosed(WindowEvent e) {
		attachWindow();
	}
    
	public void windowClosing(WindowEvent e) {
		attachWindow();
	}
	
	public void attachWindow(){
		mAttachDetachButton.setText("Detach");
		
		mManager.attachPlot(this);
	
		mDetachedFrame.remove(this);
		mDetachedFrame.setVisible(false);
	}
	
	public void windowActivated(WindowEvent e) {}
	public void windowDeactivated(WindowEvent e) {}
    public void windowDeiconified(WindowEvent e) {}
    public void windowIconified(WindowEvent e) {}
    public void windowOpened(WindowEvent e) {}

}
