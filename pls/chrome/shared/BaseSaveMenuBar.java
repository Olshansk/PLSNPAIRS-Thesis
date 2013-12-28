package pls.chrome.shared;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.MissingResourceException;

import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;

@SuppressWarnings("serial")
public class BaseSaveMenuBar extends BaseMenuBar {

	public String fileName = null;
	
	protected FileFilter filter = null;
	protected JFileChooser sChooser;
	private JFileChooser lChooser;

	protected String extension = null;
	protected String filePath = ".";


	public BaseSaveMenuBar(JFrame frame) {
		
		super(frame);
		createFileMenu();
	}
	
	public void setFileFilter(FileFilter filter, String extension) {
		this.filter = filter;
		this.extension = extension;
	}

	/**
	 * Extending class should override this to implement loading.
	 */
	public void load() {
		JOptionPane.showMessageDialog(null, 
				"Load not implemented",
				"Load from file", JOptionPane.PLAIN_MESSAGE);
	}
	
	/**
	 * Extending class should override this to implement saving.
	 */
	public void save() {
		JOptionPane.showMessageDialog(null, 
				"Save not implemented",
				"Save to file", JOptionPane.PLAIN_MESSAGE);
	}

	public void saveAs() {
		if(sChooser == null){
			sChooser = new JFileChooser(filePath);
			sChooser.setPreferredSize(new Dimension(640,480));
		}

		if(filter != null) {
			sChooser.setFileFilter(filter);
		}
		int option = sChooser.showDialog(BaseSaveMenuBar.this, "Save As");
		if(option == JFileChooser.APPROVE_OPTION) {
			fileName = sChooser.getSelectedFile().getAbsolutePath();
		//	fileName = fileName+chooser.getSelectedFile().getAbsolutePath();
			if(extension != null && !fileName.endsWith(extension)) {
				fileName += extension;
			}
			_save();
		}
		
	}
	
	private void _load() {
		if(lChooser == null){
			lChooser = new JFileChooser(filePath);
			lChooser.setPreferredSize(new Dimension(640,480));
		}

		if(filter != null) {
			lChooser.setFileFilter(filter);
		}
		int option = lChooser.showDialog(BaseSaveMenuBar.this, "Load");
		if(option == JFileChooser.APPROVE_OPTION) {
			File file = lChooser.getSelectedFile();
			this.filePath = file.getParent();
			this.fileName = file.getAbsolutePath();
			load();
		}
	}
	
	private void _save() {
		//If there is no file is save, ask the user to specify 
		//the file to save to.
		try{
			if(this.fileName == null) { 
				saveAs();
			} else {
				save();
			}
		}catch(MissingResourceException e){
			//The saveAs()/save() function in sessionProfileFrame must throw
			//this exception at certain times, and until a better  
			//solution is found it needs to be caught here as well.
			//catching it here just signals that there was an error, and that
			//we should bail. normally the save function just returns so its safe
			//to do nothing here when an error is encountered.
		}
	}
	/**
	 * Create the file menu dropdown list.
	 */
	private void createFileMenu(){
		JMenu fileMenu = getMenu(0);
		JMenuItem save;
		JMenuItem load;
        JMenuItem saveAs;

		// File menu's options
        load = new JMenuItem("Load",
				new ImageIcon(this.getClass()
				.getResource("/toolbarButtonGraphics/general/Open16.gif")));

        load.setMnemonic('L');
        load.getAccessibleContext().setAccessibleDescription("Load from file");
        load.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_load();
			}
		});
        fileMenu.add(load, 0);

        save = new JMenuItem("Save",
				new ImageIcon(this.getClass()
				.getResource("/toolbarButtonGraphics/general/Save16.gif")));
        save.setMnemonic('S');
        save.getAccessibleContext().setAccessibleDescription("Save to file");
        save.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_save();
			}
		});
        fileMenu.add(save, 1);

        saveAs = new JMenuItem("Save As ...",
				new ImageIcon(this.getClass()
				.getResource("/toolbarButtonGraphics/general/SaveAs16.gif")));
        saveAs.setMnemonic('A');
        saveAs.getAccessibleContext().setAccessibleDescription("Save as file");
        saveAs.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				try{
					saveAs();
				}catch(MissingResourceException e2){}
			}
		});
        fileMenu.add(saveAs, 2);
	}
}