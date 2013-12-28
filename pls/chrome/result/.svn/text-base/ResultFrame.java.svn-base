package pls.chrome.result;

import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Toolkit;

import javax.swing.filechooser.FileFilter;
import javax.swing.JFrame;
import javax.swing.JTabbedPane;

@SuppressWarnings("serial")
/**
 * Unused class file?
 */
public class ResultFrame extends JFrame {
	
	public static final Dimension DIMENSION = new Dimension(1280, 960);

	private ResultMenuBar menuBar;

	public ResultFrame(String fileType, String fileName, FileFilter fileFilter,
			String fileExtension) {
		super(fileType + " Results");
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);

		DetachableTabbedPane tabs = new DetachableTabbedPane();
		
		menuBar = new ResultMenuBar(this, fileType, tabs, fileFilter, fileExtension);
		setJMenuBar(menuBar);
		menuBar.fileName = fileName;
		menuBar.load();

		add(tabs);

//		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
//		setSize(screen); // For gui platforms that can't handle maximize
		setSize(DIMENSION);

		setExtendedState(getExtendedState() | Frame.MAXIMIZED_BOTH);

		setVisible(true);
	}
	
	public void dispose() {
		// break circular reference
		menuBar = null;
		super.dispose();
	}
}