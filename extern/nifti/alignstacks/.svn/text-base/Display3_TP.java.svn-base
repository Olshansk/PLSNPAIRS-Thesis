package extern.nifti.alignstacks;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import ij.plugin.frame.*;
import ij.*;
import ij.gui.*;

import extern.nifti.alignstacks.align3tp.Display;

/**
 * An object of Display3_TP class is a frame with several buttons which
 * displays axial, coronal, and sagital views from one or more axial stacks.
 * Pushing a button starts a new thread to perform one of the display options. 
 * A mouse click on a view, moves the other views so that they intersect at
 * that intersection point.
 *
 * This plug-in does not handle color information.  RGB data is converted
 *   to byte.
 *
 * This plugin should work with Java 1.1.  There are many attractive
 *   functions in Platform 2, v1.3 for image processing; however, Wayne
 *   Rasband, has provided a great deal of functionality under 1.1
 *   including a beautifully structured API.  Using only 1.1 features
 *   allows more compatibility with legacy systems.
 *
 * @author J. Anthony Parker, MD PhD <J.A.Parker@IEEE.org>
 * @version 13July2004
 *
 * @see extern.nifti.alignstacks.align3tp.Display
 */
public class Display3_TP extends PlugInFrame implements ActionListener {
	final static boolean DEBUG = false;
	final static String[] option = {
		"Start over",				// 0
		"Additional stack",			// 1
		"Scale stacks (z)",			// 2
		"About Display3_TP",		// 3
		"Cross hairs on/off",		// 4
		"Cross hair style",			// 5
		"Cross hair color",			// 6
		"Hello",					// 7
		"Hello",					// 8
		"Mirror R-L (x)",			// 9
		"Mirror ant-post (y)",		// 10
		"Mirror up-down (z)",		// 11
		"Rotate axial",				// 12
		"Rotate coronal",			// 13
		"Rotate sagital",			// 14
		"Show affine"				// 15
	};

	private static Frame instance;
	private Panel panel;
	private Button[] button = new Button[option.length];
	private Display d;	// work horse

	public Display3_TP() {
		super("Display Stack(s)");
		if(DEBUG)
			IJ.debugMode = true;
		if (instance!=null) {	// another instance running
			instance.toFront();
			return;
		}
		instance = this;
		return;
	}

	public void run(String arg) {
		String s = arg.toLowerCase();
		boolean rightHanded = true;
		if(s.indexOf("left")>=0)
			rightHanded = false;
		d = new Display(rightHanded, this);
		if(s.indexOf("about")>=0) {
			d.showAbout();
			instance = null;
			dispose();
			return;
		}
		// Display object sets up images and windows
		if(d.setup()==false) {
			instance = null;
			dispose();
			return;
		}

		// set up Panel with buttons
		setLayout(new FlowLayout());
		panel = new Panel();
		panel.setLayout(new GridLayout(4, 4, 5, 5));
		for(int i=0; i<option.length; i++){
			button[i] = addButton(option[i]);
		}
		add(panel);
		pack();		// laying out this gives it a size
		GUI.center(this);
		pack();
		show();
		return;
	}

	// make button with ActionListener
	private Button addButton(String label) {
		Button b = new Button(label);
		b.addActionListener(this);
		panel.add(b);
		return b;
	}

	// button press dispatching point
	public void actionPerformed(ActionEvent e) {
		String label = e.getActionCommand();
		if (label==null) return;
		for(int index=0; index<option.length; index++) {
			if (label.equals(option[index])) {
				new RunOption(index, d);	// start thread
				break;
			}
		}
		return;
	}

	public void processWindowEvent(WindowEvent e) {
		super.processWindowEvent(e);
		if (e.getID()==WindowEvent.WINDOW_CLOSING) {
			instance = null;
			d.unlock();
		}
		return;
	}

	public void dispose() {
		d.closeAllWindows();
		super.dispose();
		if(DEBUG)
			IJ.debugMode = false;
		return;
	}

// end of Display3_TP main frame.  The following class has been made
//   inner classes of Display3_TP so that its name will be unique wrt
//   other plugins.


/**
 * An object of this class is instantiated in Display3_TP.actionPerformed
 * After a button is pushed this class provides a new service thread
 * Between threads, user may do other ImageJ commands
 */
public class RunOption extends Thread {
	private Display d;
	private int index;

	public RunOption(int index,  Display d) {
		super("Display3 TP Thread");
		this.index = index;
		this.d = d;
		setPriority(Math.max(getPriority()-2, MIN_PRIORITY));
		start();
		return;
	}

	public void run() {
		if (!d.setup()) return;		// verify view displays
		if (!d.lock()) return;	// lock images
		try {
			d.doCommand(index);
		}
		catch(OutOfMemoryError e) {
			IJ.outOfMemory("Display3_TP");
		} catch(Exception e) {
			CharArrayWriter caw = new CharArrayWriter();
			PrintWriter pw = new PrintWriter(caw);
			e.printStackTrace(pw);
			IJ.write(caw.toString());
			IJ.showStatus("");
		}
		return;
	}

}	// end of RunOption class


}	// end of the Align3_TP class

