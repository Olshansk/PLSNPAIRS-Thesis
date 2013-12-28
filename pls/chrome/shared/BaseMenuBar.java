package pls.chrome.shared;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.ImageIcon;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.KeyStroke;

import pls.shared.GlobalVariablesFunctions;

/**
 * An extension of JMenuBar which contains some of the menu items expected to
 * be found in all windows in PLS that have a menu.
 */
@SuppressWarnings("serial")
public class BaseMenuBar extends JMenuBar {
	
	private HelpDialog helpDialog = new HelpDialog("There is no help data associated with this window.");
	
	public BaseMenuBar(JFrame frame) {
		
        // Build the file menu.
        JMenu file = new JMenu("File");
        file.setMnemonic('F');
        file.getAccessibleContext().setAccessibleDescription("Main program menu");
        add(file);
        
        // File menu's options
        JMenuItem exit = new JMenuItem("Exit", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Stop16.gif")));
        exit.setMnemonic('X');
        exit.getAccessibleContext().setAccessibleDescription("Exit the program");
        exit.addActionListener(new CloseActionListener(frame));
        file.add(exit);

        // Build the help menu.
        JMenu help = new JMenu("Help");
        help.setMnemonic('H');
        help.getAccessibleContext().setAccessibleDescription("About this program and documentation");
        add(help);
        
        // Help menu's options
        JMenuItem helpContents = new JMenuItem("Help Contents", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Help16.gif")));
        helpContents.setMnemonic('C');
        helpContents.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F3, 0));
        helpContents.getAccessibleContext().setAccessibleDescription("Documentation and more");
        helpContents.addActionListener(new ActionListener() { 
        	public void actionPerformed(ActionEvent e) { 
        		displayHelpContents(); 
        	}
        });
        help.add(helpContents);
        JMenuItem about = new JMenuItem("About PLS and NPAIRS", new ImageIcon(this.getClass().
        		getResource("/toolbarButtonGraphics/general/About16.gif")));
        about.setMnemonic('A');
        about.getAccessibleContext().setAccessibleDescription("More about this program");
        about.addActionListener(
        		new ActionListener() {
        			public void actionPerformed(ActionEvent e) {
        				JOptionPane.showMessageDialog(null, "This is a Java translation and enhancement of PLS" +
        						" (originally written in Matlab) \nand NPAIRS (originally written in IDL)." +
        						"\nVersion " + GlobalVariablesFunctions.getVersion() + ".\nPlease contact Stephen Strother " +
        						"<sstrother@rotman-baycrest.on.ca> \nor Anita Oder <aoder@rotman-baycrest.on.ca> " +
        						"to report any bugs or make comments, \nor visit our open-source project at: " +
        						"<http://code.google.com/p/plsnpairs/>.", "About PLS and NPAIRS", JOptionPane.INFORMATION_MESSAGE);
        				}
        			}
        		);
        help.add(about);
	}
	
	public void setHelpData(String helpData) {
		helpDialog.setHelpData(helpData);
	}
	
	private void displayHelpContents() {
		if(!helpDialog.isVisible()) {
			helpDialog.setVisible(true);
		} else {
			helpDialog.toFront();
		}
	}
}

@SuppressWarnings("serial")
final class HelpDialog extends JFrame  {
	
	private HelpDialog dialog = null;
	
	private JTextArea textArea = null;
	
	public HelpDialog(String helpData) {
		this.dialog = this;
		
		setDefaultCloseOperation(JDialog.HIDE_ON_CLOSE);
		
		textArea = new JTextArea(helpData + "\n\nFor more help with using PLS and NPAIRS, " +
				"\nvisit http://code.google.com/p/plsnpairs/.");
		textArea.setFont(new Font("Courier", Font.PLAIN, 12));
		textArea.setEditable(false);
		textArea.setBorder(BorderFactory.createEmptyBorder(2,2,2,2));
	    
	    JScrollPane textScroller = new JScrollPane(textArea);
	    textScroller.setPreferredSize(new Dimension(400, 150));
		
		JButton button = new JButton("Ok");
		button.setIconTextGap(15);
	    button.addActionListener(new ActionListener() {
	        public void actionPerformed(ActionEvent e) {
	            dialog.setVisible(false);
	        }
	    });
	    
	    JPanel mainPane = new JPanel(new BorderLayout(10, 10));

	    JPanel topPane = new JPanel(new BorderLayout());
	    topPane.add(textScroller, BorderLayout.NORTH);
	    
	    JPanel bottomPane = new JPanel(new BorderLayout(5, 5));
	    bottomPane.add(button, BorderLayout.SOUTH);
	    
	    mainPane.add(topPane, BorderLayout.NORTH);
	    mainPane.add(bottomPane, BorderLayout.SOUTH);
	    
	    mainPane.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
	    
	    add(mainPane);
	    
	    setTitle("PLS Help Contents");
	    pack();
        // Position the dialog on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        setResizable(false);
	}
	
	public void setHelpData(String helpData) {
		textArea.setText(helpData + "\n\nFor more help with using PLS and NPAIRS, " +
				"\nvisit http://code.google.com/p/plsnpairs/.");
	}
}

final class CloseActionListener implements ActionListener {
	
	private JFrame frame = null;
	
	public CloseActionListener(JFrame frame) {
		this.frame = frame;
	}
	public void actionPerformed(ActionEvent e) {
		frame.dispose();
	}
}
