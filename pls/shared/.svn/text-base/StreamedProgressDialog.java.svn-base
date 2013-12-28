package pls.shared;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;

import pls.chrome.shared.ProgressDialogWatcher;

@SuppressWarnings("serial")
public class StreamedProgressDialog extends JDialog implements ActionListener {

	private JTextPane mTextPane = null;
	
	private JProgressBar mProgressBar = null;
	
	private JButton mMainButton = null;
	
	private PipedInputStream mInput = null;
	
	private boolean mIsComplete = false;
	
	public ProgressDialogWatcher worker = null;
	
	/**
	 * Creates a new StreamedProgressDialog.
	 * @param maxSize The maximum number of ticks on the progress bar.
	 */
	public StreamedProgressDialog(JFrame parent, int maxSize) {
		super(parent);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		setTitle("Processing task...");
		
		mTextPane = new JTextPane();
		mTextPane.setEditable(false);
		
		JPanel southPane = new JPanel(new BorderLayout() );
		
		mProgressBar = new JProgressBar(0, maxSize);
		mProgressBar.setPreferredSize(new Dimension(mProgressBar.getPreferredSize().width, 20) );
		
		mMainButton = new JButton("Stop", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Stop16.gif")));
		mMainButton.setIconTextGap(15);
		
		southPane.add(mProgressBar);
		southPane.add(mMainButton, BorderLayout.SOUTH);
		
		add(new JScrollPane(mTextPane));
		add(southPane, BorderLayout.SOUTH);
		
		mInput = new PipedInputStream();
		
		setSize(640, 480);
		setVisible(true);
		
		mMainButton.addActionListener(this);
	}
	
	// In a loop, waits for input from its PipedInputStream.
	// When the Stream is closed, stops the thread.
	private void listenToReader() throws IOException {
		InputStreamReader reader = new InputStreamReader(mInput);
		
		char[] buffer = new char[128];
		
		int i = reader.read(buffer);
		
		while (i > 0) {
			mProgressBar.setValue(mProgressBar.getValue() + 1);
			
			String message = new String(buffer, 0, i);
			
			mTextPane.setText(mTextPane.getText() + message);
			mTextPane.setCaretPosition(mTextPane.getDocument().getLength());

			buffer = new char[128];
			i = reader.read(buffer);
		}
		
		complete();
	}
	
	public void complete() {
//		mTextPane.setText(mTextPane.getText() + "Approx number of ticks: " + mProgressBar.getValue() );
		mProgressBar.setMaximum(mProgressBar.getValue() );
		mIsComplete = true;
		
		setTitle("Complete");
		
		mMainButton.setText("Ok");
		mMainButton.setIcon(null);
		//dispose();
	}
	
	public boolean isComplete() {
		return mIsComplete;
	}
	
	// Connects the dialog to a PipedOutputStream
	public void connectWriter(PipedOutputStream outputStream) throws IOException {
		mInput.connect(outputStream);
		
		new Thread() {
			public void run() {
				try {
					listenToReader();
				} catch (IOException e) {
					System.err.println("Thread providing input to the dialog was terminated.");
//					e.printStackTrace();
					try {
						mInput.close();
					} catch (IOException e1) {
//						e1.printStackTrace();
					}
				}
			}
		}.start();
	}

	/**
	 * Listens for the stop and ok buttons.
	 */
	@Override
	public void actionPerformed(ActionEvent arg0) {
		if (mMainButton.getText().equals("Stop")) {
        	JOptionPane confirmDialog = createConfirmDialog();
        	if (confirmDialog.getValue() != null) {
				int option = ((Integer) confirmDialog.getValue()).intValue();
    			if (option == JOptionPane.YES_OPTION) {
	            	dispose();
        		}
			}
    	} else {
    		dispose();
        }
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
		/** TODO **/
		/// WARNING!  By using worker.stop() instead of worker.interrupt(),
		/// we prevent each individual worker from having to deal with checking
		/// for interrupts.  This works out ok since none of our threads are
		/// cooperative.  However, this could result in deadlocks in certain
		/// situations if we had collaborative threads going.
		if (worker != null) {
			worker.stop();
		}
		worker = null;
		
		try {
			mInput.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		super.dispose();
	}
	
//	public static void main(String[] argv) throws InterruptedException, IOException {
//	StreamedProgressDialog dialog = new StreamedProgressDialog(null, 32);
//	dialog.setSize(640, 480);
//	dialog.setVisible(true);
//	
//	PipedOutputStream pos = new PipedOutputStream();
//	dialog.connectWriter(pos);
//	
//	File file = new File("demo_log.txt");
//	
//	FileOutputStream fileos = new FileOutputStream(file);
//	
//	StreamedProgressHelper stream = new StreamedProgressHelper();
//	stream.addStream(pos);
//	stream.addStream(System.out);
//	stream.addStream(fileos);
//	
//	SomeRoutine(stream);
//	
//	Thread.sleep(2000);
//	
//	System.exit(0);
//}

	// This is some function that prints out its progress.  Does it do it to
	// a progress dialog?  The console?  A file?  Who knows!
	private static void SomeRoutine(StreamedProgressHelper stream) throws InterruptedException {
		stream.startTask("Streaming Progress Demo", "Demo");
		
		stream.startTask("Initializing something or other.");
		
		Thread.sleep(2800);
		
		stream.endTask();
		
		stream.startTask("Performing analysis", "Analysis");
		
		for (int i = 1; i <= 3; ++i) {
			Thread.sleep(600);
			stream.startTask("Starting subroutine " + i);
			Thread.sleep((long)(Math.random() * 1000.0 + 1000));
			stream.endTask();
		}
		
		stream.startTask("Starting subroutine 4");
		
		Thread.sleep(400);
		stream.startTask("Part one");
		Thread.sleep(400);
		stream.endTask();
		stream.startTask("Part two");
		Thread.sleep(400);
		stream.endTask();
		stream.startTask("Part three");
		Thread.sleep(400);
		stream.endTask();
		
		// Ending subroutine 4
		Thread.sleep(400);
		stream.endTask();
		
		for (int i = 5; i <= 9; ++i) {
			Thread.sleep(200);
			stream.startTask("Starting subroutine " + i);
			Thread.sleep((long)(Math.random() * 500.0 + 500));
			stream.endTask();
		}
		
		// Ending the analysis
		stream.endTask();
		
		Thread.sleep(1300);
		
		// Ending the demo
		stream.endTask();
		
		stream.complete();
		
		Thread.sleep(10000);
	}
}
