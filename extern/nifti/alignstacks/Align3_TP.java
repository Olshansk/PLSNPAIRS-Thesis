package extern.nifti.alignstacks;
import extern.nifti.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*; 
import ij.plugin.frame.*;
import ij.*;
import ij.gui.*;

import extern.nifti.alignstacks.align3tp.Alignment;

/**
 * This ImageJ plugin allows the user to align two source stacks.
 * The source stacks are the first and second open stacks in the "Window"
 *   menu.
 * The alignment may optionally be applied to a third stack, the current
 *   stack.  For example, a PET transmission scan could be aligned with a
 *   CT, and the alignment applied to a PET emission scan.
 *
 * An object of Align3_TP class is a frame with several buttons.
 *   Pushing a button starts a new thread to perform the task.
 *
 * This plug-in does not handle color information.  RGB data is converted
 *   to byte.  Note however that output stacks can be colored and then
 *   combined using ImageJ options.
 *
 * This plugin should work with Java 1.1.  There are many attractive
 *   functions in Platform 2, v1.3 for image processing; however, Wayne
 *   Rasband, has provided a great deal of functionality under 1.1
 *   including a beautifully structured API.  Using only 1.1 features
 *   allows more compatibility with legacy systems.
 *
 * @author J. Anthony Parker, MD PhD <J.A.Parker@IEEE.org>
 * @version 10November2004
 *
 * @see extern.nifti.alignstacks.align3tp.Alignment
 */
public class Align3_TP extends PlugInFrame implements ActionListener {
	final static String[] option = {
		"Start over",			// 0
		"Layout images",		// 1
		"Undo",					// 2
		"History in/out",		// 3
		"Align slices",			// 4
		"Match ROIs",			// 5
		"Shear horizontal",		// 6
		"Shear vertical",		// 7
		"Mouse translate",		// 8
		"Mouse rotate",			// 9
		"Mouse resize",			// 10
		"Mouse cancel",			// 11
		"Resample along line",	// 12
		"Axial (x,y)",			// 13
		"Coronal (x,z)",		// 14
		"Sagital (y,z)",		// 15
		"Scale stacks (z)",		// 16
		"Rotate view",			// 17
		"Mirror horizontal",	// 18
		"Mirror vertical",		// 19
		"Increment slice",		// 20
		"Decrement slice",		// 21
		"Goto slice",			// 22
		"Goto again",			// 23
		"Show 3D region",		// 24
		"Set 3D region",		// 25
		"Slice registration",	// 26
//		"Hello",				// 26
		"Volume registration",	// 27
//		"Hello",				// 27
		"About Align3_TP",		// 28
		"Output help",			// 29
		"Output",				// 30
		"Show affine"			// 31
	};
	final static String[] optionE = {
		"About external",		// 32
		"Show external data",	// 33
		"Save data",			// 34
		"Retrieve data"			// 35
	};

	private static Frame instance;
	private Alignment a;		// The Alignment class is the work horse
	private Panel panel;
	private Button[] button = new Button[option.length+optionE.length];

	public Align3_TP() {
		super("Align Stacks");

		if (instance!=null) {	// another instance running
			instance.toFront();
			return;
		}
		instance = this;
		return;
	}

	public void run(String arg) {
		String s = arg.toLowerCase();
		boolean rightHanded = true, external = false;
		if(s.indexOf("left")>=0)
			rightHanded = false;
		if(s.indexOf("external")>=0)
			external = true;
		a = new Alignment(rightHanded, this);
		if(s.indexOf("about")>=0) {
			a.showAbout();
			instance = null;
			dispose();
			return;
		}
		// setup Display class and images and windows
		if(a.setup()==false) {
			instance = null;
			dispose();
			return;
		}

		// set up Panel with buttons
		setLayout(new FlowLayout());
		panel = new Panel();
		if(external)
			panel.setLayout(new GridLayout(9, 4, 5, 5));
		else
			panel.setLayout(new GridLayout(8, 4, 5, 5));
		for(int i=0; i<option.length; i++)
			button[i] = addButton(option[i]);
		if(external)
			for(int i=0; i<optionE.length; i++)
				button[option.length+i] = addButton(optionE[i]);
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
				new RunOption(index, a);	// start thread
				return;
			}
		}
		for(int index=0; index<optionE.length; index++) {
			if (label.equals(optionE[index])) {
				new RunOption(option.length+index, a);	// start thread
				return;
			}
		}
		return;
	}

	public void processWindowEvent(WindowEvent e) {
		super.processWindowEvent(e);
		if (e.getID()==WindowEvent.WINDOW_CLOSING) {
			instance = null;
			a.unlock();
		}
		return;
	}

	public void dispose() {
		a.closeWindows();
		super.dispose();
		return;
	}

// end of Align3_TP main frame.  The following class has been made
//   inner classes of Align3_TP so that its name will be unique wrt
//   other plugins.


/**
 * An object of this class is instantiated in Align3_TP.actionPerformed
 * After a button is pushed this class provides a new service thread
 * Between threads, user may do other ImageJ commands
 */
public class RunOption extends Thread {
	private Alignment a;
	private int index;

	public RunOption(int index,  Alignment a) {
		super("Align3TP Thread");
		this.index = index;
		this.a = a;
		setPriority(Math.max(getPriority()-2, MIN_PRIORITY));
		start();
		return;
	}

	public void run() {
		if (!a.setup()) return;		// verify image window display
		if(index!=11)
			if (!a.lock()) return;	// lock images
		try {
			a.doCommand(index);
		}
		catch(OutOfMemoryError e) {
			IJ.outOfMemory("Align3_TP");
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

