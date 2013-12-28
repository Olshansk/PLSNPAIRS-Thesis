package pls.chrome.shared;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.DecimalFormat;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JOptionPane;
import javax.swing.JTextArea;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;

import nl.jj.swingx.gui.modal.JModalFrame;

@SuppressWarnings("serial")
public class ProgressDialog extends JModalFrame  {
	
	private ProgressDialog dialog = null;
	
	private JTextArea textArea = null;
	
	private JProgressBar progressBar = null;
	
	protected JPanel buttonPane = null;
	
	private JButton mainButton = null;
	
	private ProgressDialogWatcher worker = null;
	
	private double startTime = 0;
	
	private double timeSoFar = 0;
	
	// This variable set to volatile for multi-threading purposes
	private volatile boolean isComplete = false;
	
	public ProgressDialog(final JFrame parent, int maxSize, ProgressDialogWatcher worker) {
		this.dialog = this;
		this.worker = worker;
		
		setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		//addAdditionalModalToWindow(parent);
		
		textArea = new JTextArea("Processing Task...  (0 seconds)");
		textArea.setFont(new Font("Courier", Font.PLAIN, 12));
		textArea.setEditable(false);
		textArea.setBorder(BorderFactory.createEmptyBorder(2,2,2,2));
	    
		JScrollPane textScroller = new JScrollPane(textArea);
	    textScroller.setPreferredSize(new Dimension(400, 150));
		
		progressBar = new JProgressBar(0, maxSize);
		progressBar.setPreferredSize(new Dimension(200, 20));
		
//		worker.setProgressDialog(this);
		startTime = System.currentTimeMillis();
		timeSoFar = startTime;
		worker.start();
		
		mainButton = new JButton("Stop", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Stop16.gif")));
		mainButton.setIconTextGap(15);
	    mainButton.addActionListener(new ActionListener() {
	        public void actionPerformed(ActionEvent e) {
	        	if (mainButton.getText().equals("Stop")) {
	            	JOptionPane confirmDialog = createConfirmDialog();
	            	if (confirmDialog.getValue() != null) {
	    				int option = ((Integer) confirmDialog.getValue()).intValue();
		    			if (option == JOptionPane.YES_OPTION) {
		        			dialog.dispose();
			            	parent.dispose();
		        		}
	    			}
	        	} else {
	        		dialog.dispose();
	            }
	        }
	    });
	    
	    JPanel mainPane = new JPanel(new BorderLayout(10, 10));

	    JPanel topPane = new JPanel(new BorderLayout());
	    topPane.add(textScroller, BorderLayout.NORTH);
	    
	    JPanel bottomPane = new JPanel(new BorderLayout(5, 5));
	    bottomPane.add(progressBar, BorderLayout.NORTH);
	    
	    buttonPane = new JPanel(new GridLayout(1, 2, 0, 0));
	    buttonPane.add(mainButton);
	    bottomPane.add(buttonPane, BorderLayout.SOUTH);
	    
	    mainPane.add(topPane, BorderLayout.NORTH);
	    mainPane.add(bottomPane, BorderLayout.SOUTH);
	    
	    mainPane.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
	    
	    add(mainPane);
	    
	    setTitle("Processing Task");
	    pack();
        // Position the dialog on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        setResizable(false);
	    setVisible(true);
	    toFront();
	    setAlwaysOnTop(true);
	}
	
	private JOptionPane createConfirmDialog() {
		JOptionPane optionPane = new JOptionPane("Are you sure you want to stop the process?",
				   JOptionPane.WARNING_MESSAGE,
				   JOptionPane.YES_NO_OPTION);

		JDialog confirmDialog = optionPane.createDialog(null, "Stop Process");
		confirmDialog.setAlwaysOnTop(true);
		confirmDialog.setVisible(true);

		return optionPane;
	}
	
	public void dispose() {
	//	System.out.println("Disposin, yeah!");
    	//worker.interrupt();
		/** TODO **/
		/// WARNING!  By using worker.stop() instead of worker.interrupt(),
		/// we prevent each individual worker from having to deal with checking
		/// for interrupts.  This works out ok since none of our threads are
		/// cooperative.  However, this could result in deadlocks in certain
		/// situations if we had collaborative threads going.
		worker.stop();
		worker = null;
		super.dispose();
	}
	
	public void complete() {
		isComplete = true;
		setTitle("Complete");
		progressBar.setValue(progressBar.getMaximum());
		setFinalMessage("\nThe task has been completed.\nPlease check your results.");
		toFront();
	}
	
	public void error(Throwable ex) {
		setTitle("Error");
		String msg = ex.getMessage();
		if(msg == null) {
			msg = "Something horrible went wrong.";
		}
		StringWriter sw = new StringWriter();
		ex.printStackTrace(new PrintWriter(sw));
		setFinalMessage("\n\nError Message:\n" + msg + "\n\nStack Trace:\n" + sw.toString());
		Toolkit.getDefaultToolkit().beep();
		toFront();
	}
	
	public void appendMessage(String message) {
		timeSoFar = System.currentTimeMillis();
		textArea.append("\n- " + message);
		textArea.setCaretPosition(textArea.getDocument().getLength());
	}
	
	public void updateStatus(int increment) {
		double currentTime = System.currentTimeMillis();
		String formattedTime = new DecimalFormat("###.###").format((currentTime - timeSoFar) / 1000.0);
		textArea.append(" (" + formattedTime + " seconds)");
		textArea.setCaretPosition(textArea.getDocument().getLength());
		progressBar.setValue(progressBar.getValue() + increment);
		timeSoFar = currentTime;
	}
	
	public void updateStatus(String message, int increment) {
		double currentTime = System.currentTimeMillis();
		String formattedTime = new DecimalFormat("###.###").format((currentTime - timeSoFar) / 1000.0);
		textArea.append("\n- " + message + " (" + formattedTime + " seconds)");
		textArea.setCaretPosition(textArea.getDocument().getLength());
		progressBar.setValue(progressBar.getValue() + increment);
		timeSoFar = currentTime;
	}
	
	public void setFinalMessage(String message) {
		progressBar.setValue(progressBar.getMaximum());
		double totalTime = (System.currentTimeMillis() - startTime) / 1000.0;
		String formattedTime = new DecimalFormat("###.###").format(totalTime);
		textArea.append(message + "\nTotal task execution time: " + formattedTime + " seconds.");
		textArea.setCaretPosition(textArea.getDocument().getLength());
		mainButton.setText("Ok");
		mainButton.setIcon(null);
	}
	
	public boolean isComplete() {
		return isComplete;
	}
}
